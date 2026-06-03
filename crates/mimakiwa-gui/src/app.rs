use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering::Relaxed};

use iced::{
    widget::{button, checkbox, column, container, horizontal_rule, radio, row, scrollable,
             slider, svg, text, text_input, Space},
    Alignment, Background, Border, Color, Element, Length, Padding, Task, Theme,
    theme,
};

use mimakiwa_model::MimakiwaModel;
use mimakiwa_tokenizer::BPETokenizer;
use mimakiwa_self_learn::{SealLearner, SealConfig};
use mimakiwa_search::KnowledgeStore;

use crate::theme as col;

// ── Global model/tokenizer store ──────────────────────────────────────────────

struct Loaded {
    model: MimakiwaModel,
    tokenizer: BPETokenizer,
}

static LOADED: std::sync::LazyLock<Mutex<Option<Loaded>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

// Training progress atomics (written by bootstrap thread, read by GUI tick)
static PROG_STAGE: AtomicU8    = AtomicU8::new(0);  // 1=dl 2=bpe 3=train 4=save
static PROG_STEP:  AtomicUsize = AtomicUsize::new(0);
static PROG_TOTAL: AtomicUsize = AtomicUsize::new(0);
static PROG_LOSS:  AtomicU32   = AtomicU32::new(0);  // f32 bits
static PROG_TOKS:  AtomicU32   = AtomicU32::new(0);  // f32 bits (tok/s)
static PROG_START: AtomicUsize = AtomicUsize::new(0); // ms timestamp

// ── Dataset presets ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct DatasetPreset {
    name: &'static str,
    desc: &'static str,
    url: &'static str,
    format: &'static str,
    max_mb: usize,
}

const DATASETS: &[DatasetPreset] = &[
    DatasetPreset {
        name: "TinyStories",
        desc: "20 MB · Coherent English stories · Best for small models · ★ Recommended",
        url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        format: "plain",
        max_mb: 20,
    },
    DatasetPreset {
        name: "Dolly 15k",
        desc: "3 MB · Instruction-following Q&A · Good dialogue quality",
        url: "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        format: "dolly_jsonl",
        max_mb: 5,
    },
    DatasetPreset {
        name: "WikiText-103",
        desc: "10 MB · Factual Wikipedia text · Good for knowledge questions",
        url: "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw/wiki.train.raw",
        format: "plain",
        max_mb: 10,
    },
];

// ── Model size ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSize {
    Compact,  // ~15M params
    Small,    // ~85M params
    Medium,   // ~350M params
}

impl ModelSize {
    fn label(self) -> &'static str {
        match self {
            ModelSize::Compact => "Compact (~15M params) · ~20 min · ★ Recommended",
            ModelSize::Small   => "Small   (~85M params) · ~60 min · Better quality",
            ModelSize::Medium  => "Medium  (~350M params) · ~4 hrs  · Best quality",
        }
    }

    fn steps(self) -> usize {
        match self {
            ModelSize::Compact => 20_000,
            ModelSize::Small   => 40_000,
            ModelSize::Medium  => 80_000,
        }
    }

    fn seq_len(self) -> usize {
        match self {
            ModelSize::Compact => 256,
            ModelSize::Small   => 512,
            ModelSize::Medium  => 1024,
        }
    }

    fn batch_size(self) -> usize {
        match self {
            ModelSize::Compact => 8,
            ModelSize::Small   => 4,
            ModelSize::Medium  => 2,
        }
    }
}

// ── Bootstrap config ──────────────────────────────────────────────────────────

struct BootstrapConfig {
    dataset_url:    String,
    dataset_format: String,
    dataset_max_mb: usize,
    model_size:     ModelSize,
    vocab_size:     usize,
    training_steps: usize,
    batch_size:     usize,
    seq_len:        usize,
    learning_rate:  f32,
}

// ── Bootstrap result ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum BootstrapResult {
    Ok { params: usize, vocab: usize },
    Err(String),
}

fn mimakiwa_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".mimakiwa")
}

// ── Data types ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role:       Role,
    pub content:    String,
    pub confidence: Option<f32>,
    pub sources:    Vec<String>,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Role { User, Assistant, System }

#[derive(Clone, Debug)]
struct SessionEntry {
    title:    String,
    messages: Vec<ChatMessage>,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub temperature:      f32,
    pub top_p:            f32,
    pub max_tokens:       String,
    pub enable_research:  bool,
    pub enable_seal:      bool,
    pub show_confidence:  bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            temperature:     0.8,
            top_p:           0.9,
            max_tokens:      "200".to_string(),
            enable_research: true,
            enable_seal:     true,
            show_confidence: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum View { DatasetSelect, Training, Chat, Settings }

// ── App ───────────────────────────────────────────────────────────────────────

pub struct MimakiwaApp {
    model:     Option<Arc<Mutex<MimakiwaModel>>>,
    tokenizer: Option<Arc<BPETokenizer>>,
    seal:      Option<Arc<Mutex<SealLearner>>>,

    messages:      Vec<ChatMessage>,
    past_sessions: Vec<SessionEntry>,
    input:         String,
    is_generating: bool,
    history_ids:   Vec<u32>,

    settings:      Settings,
    view:          View,
    model_status:  String,
    loading_stage: String,
    sidebar_open:  bool,

    selected_dataset: usize,
    custom_url:       String,
    selected_size:    ModelSize,
}

// ── Messages ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Msg {
    SelectDataset(usize),
    SetCustomUrl(String),
    SelectSize(ModelSize),
    StartTraining,
    TrainTick,
    TrainingDone(BootstrapResult),

    TypeInput(String),
    Send,
    Generated(GenResult),

    OpenSettings,
    CloseSettings,
    SetTemperature(f32),
    SetTopP(f32),
    SetMaxTokens(String),
    ToggleResearch(bool),
    ToggleSeal(bool),
    ToggleConfidence(bool),

    NewChat,
    RestoreSession(usize),
    ToggleSidebar,
    OpenUrl(String),
    RetrainModel,
}

#[derive(Clone, Debug)]
pub enum GenResult {
    Ok { text: String, confidence: f32, sources: Vec<String>, history_ids: Vec<u32>, elapsed_ms: u64 },
    Err(String),
}

// ── impl ──────────────────────────────────────────────────────────────────────

impl MimakiwaApp {
    pub fn new() -> (Self, Task<Msg>) {
        let dir = mimakiwa_dir();
        let model_path = dir.join("mimakiwa.bin");
        let tok_path   = dir.join("tokenizer.json");

        let (view, loading_stage, initial_task) = if model_path.exists() && tok_path.exists() {
            let task = Task::perform(async move {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    tx.send(load_existing(&model_path, &tok_path)).ok();
                });
                rx.recv().unwrap_or(BootstrapResult::Err("Thread error".into()))
            }, Msg::TrainingDone);
            (View::Training, "Loading saved model…".into(), task)
        } else {
            (View::DatasetSelect, String::new(), Task::none())
        };

        (Self {
            model:     None,
            tokenizer: None,
            seal:      None,
            messages:  vec![],
            past_sessions: vec![],
            input:     String::new(),
            is_generating: false,
            history_ids: vec![],
            settings:  Settings::default(),
            view,
            model_status:  String::new(),
            loading_stage,
            sidebar_open:  false,
            selected_dataset: 0,
            custom_url:  String::new(),
            selected_size: ModelSize::Compact,
        }, initial_task)
    }

    pub fn theme(&self) -> Theme {
        Theme::custom("Mimakiwa".to_string(), theme::Palette {
            background: col::BG_MAIN,
            text:       col::TEXT,
            primary:    col::ACCENT,
            success:    col::GREEN,
            danger:     Color::from_rgb(0.85, 0.22, 0.22),
        })
    }

    pub fn subscription(&self) -> iced::Subscription<Msg> {
        if self.view == View::Training {
            iced::time::every(std::time::Duration::from_millis(200))
                .map(|_| Msg::TrainTick)
        } else {
            iced::Subscription::none()
        }
    }

    pub fn update(&mut self, msg: Msg) -> Task<Msg> {
        match msg {
            Msg::SelectDataset(i) => { self.selected_dataset = i; Task::none() }
            Msg::SetCustomUrl(s)  => { self.custom_url = s; Task::none() }
            Msg::SelectSize(s)    => { self.selected_size = s; Task::none() }

            Msg::StartTraining => {
                let (url, format, max_mb) = if self.custom_url.trim().is_empty() {
                    let d = &DATASETS[self.selected_dataset];
                    (d.url.to_string(), d.format.to_string(), d.max_mb)
                } else {
                    (self.custom_url.trim().to_string(), "plain".to_string(), 20)
                };

                let cfg = BootstrapConfig {
                    dataset_url:    url,
                    dataset_format: format,
                    dataset_max_mb: max_mb,
                    model_size:     self.selected_size,
                    vocab_size:     8192,
                    training_steps: self.selected_size.steps(),
                    batch_size:     self.selected_size.batch_size(),
                    seq_len:        self.selected_size.seq_len(),
                    learning_rate:  3e-4,
                };

                self.view = View::Training;
                self.loading_stage = "Starting…".into();
                PROG_STAGE.store(0, Relaxed);
                PROG_STEP.store(0, Relaxed);
                PROG_TOTAL.store(cfg.training_steps, Relaxed);
                PROG_START.store(unix_now_ms(), Relaxed);

                Task::perform(async move {
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        tx.send(bootstrap(cfg)).ok();
                    });
                    rx.recv().unwrap_or(BootstrapResult::Err("Thread error".into()))
                }, Msg::TrainingDone)
            }

            Msg::TrainTick => {
                let stage = PROG_STAGE.load(Relaxed);
                let step  = PROG_STEP.load(Relaxed);
                let total = PROG_TOTAL.load(Relaxed);
                let loss  = f32::from_bits(PROG_LOSS.load(Relaxed));
                let toks  = f32::from_bits(PROG_TOKS.load(Relaxed));
                let start = PROG_START.load(Relaxed) as u64;

                self.loading_stage = match stage {
                    1 => format!("Downloading dataset…  {step} KB"),
                    2 => "Building vocabulary (BPE tokenizer)…".into(),
                    3 if total > 0 => {
                        let pct = step as f32 / total as f32 * 100.0;
                        let eta = if toks > 0.0 && step > 0 {
                            let elapsed_s = (unix_now_ms() as u64).saturating_sub(start) as f32 / 1000.0;
                            let steps_left = total.saturating_sub(step);
                            let secs_per_step = elapsed_s / step as f32;
                            let eta_s = steps_left as f32 * secs_per_step;
                            if eta_s < 60.0 { format!("{:.0}s", eta_s) }
                            else { format!("{:.0}m", eta_s / 60.0) }
                        } else { "…".into() };
                        format!(
                            "Training  {step}/{total}  ({pct:.0}%)  loss {loss:.3}  {toks:.0} tok/s  ETA {eta}"
                        )
                    }
                    4 => "Saving model…".into(),
                    _ => self.loading_stage.clone(),
                };
                Task::none()
            }

            Msg::TrainingDone(r) => {
                match r {
                    BootstrapResult::Ok { params, vocab } => {
                        if let Some(loaded) = LOADED.lock().unwrap().take() {
                            let tok = Arc::new(loaded.tokenizer);
                            let mdl = Arc::new(Mutex::new(loaded.model));
                            self.tokenizer = Some(tok);
                            self.model     = Some(mdl);
                            if self.settings.enable_seal {
                                self.seal = Some(Arc::new(Mutex::new(
                                    SealLearner::new(SealConfig::default())
                                )));
                            }
                            self.model_status = format!("{:.1}M params · {}k vocab", params as f32 / 1e6, vocab / 1000);
                        }
                        self.view = View::Chat;
                        self.push(Role::System,
                            format!("Mimakiwa ready — {} · SEAL on · web research on", self.model_status),
                            None, vec![], 0);
                    }
                    BootstrapResult::Err(e) => {
                        self.loading_stage = format!("Error: {e}");
                    }
                }
                Task::none()
            }

            Msg::TypeInput(s) => { self.input = s; Task::none() }

            Msg::Send => {
                let raw = self.input.trim().to_string();
                if raw.is_empty() || self.is_generating { return Task::none(); }
                self.input.clear();

                if raw.starts_with('/') || raw.starts_with('#') {
                    self.run_command(&raw);
                    return Task::none();
                }

                self.push(Role::User, raw.clone(), None, vec![], 0);
                self.is_generating = true;

                let model     = self.model.clone().unwrap();
                let tokenizer = self.tokenizer.clone().unwrap();
                let seal      = self.seal.clone();
                let cfg       = self.settings.clone();
                let hist      = self.history_ids.clone();

                let gen = Task::perform(async move {
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        tx.send(generate(raw, hist, model, tokenizer, seal, cfg)).ok();
                    });
                    rx.recv().unwrap_or(GenResult::Err("Thread error".into()))
                }, Msg::Generated);
                let scroll = scrollable::snap_to(scroll_id(), scrollable::RelativeOffset::END);
                Task::batch(vec![gen, scroll])
            }

            Msg::Generated(r) => {
                self.is_generating = false;
                match r {
                    GenResult::Ok { text, confidence, sources, history_ids, elapsed_ms } => {
                        self.history_ids = history_ids;
                        let conf = self.settings.show_confidence.then_some(confidence);
                        self.push(Role::Assistant, text, conf, sources, elapsed_ms);
                    }
                    GenResult::Err(e) => self.push(Role::System, e, None, vec![], 0),
                }
                scrollable::snap_to(scroll_id(), scrollable::RelativeOffset::END)
            }

            Msg::NewChat => {
                if !self.messages.is_empty() {
                    let title = self.messages.iter()
                        .find(|m| m.role == Role::User)
                        .map(|m| m.content.chars().take(32).collect::<String>() + "…")
                        .unwrap_or("Chat".into());
                    self.past_sessions.push(SessionEntry {
                        title,
                        messages: std::mem::take(&mut self.messages),
                    });
                }
                self.history_ids.clear();
                Task::none()
            }

            Msg::RestoreSession(i) => {
                if let Some(s) = self.past_sessions.get(i) { self.messages = s.messages.clone(); }
                self.sidebar_open = false;
                self.history_ids.clear();
                Task::none()
            }

            Msg::ToggleSidebar => { self.sidebar_open = !self.sidebar_open; Task::none() }

            Msg::OpenUrl(url) => {
                #[cfg(target_os = "macos")]
                { let _ = std::process::Command::new("open").arg(&url).spawn(); }
                Task::none()
            }

            Msg::RetrainModel => {
                // Delete saved model and go back to DatasetSelect
                let dir = mimakiwa_dir();
                let _ = std::fs::remove_file(dir.join("mimakiwa.bin"));
                let _ = std::fs::remove_file(dir.join("mimakiwa.cfg.json"));
                let _ = std::fs::remove_file(dir.join("tokenizer.json"));
                self.model     = None;
                self.tokenizer = None;
                self.seal      = None;
                self.messages.clear();
                self.history_ids.clear();
                self.loading_stage.clear();
                PROG_STAGE.store(0, Relaxed);
                self.view = View::DatasetSelect;
                Task::none()
            }

            Msg::OpenSettings  => { self.view = View::Settings; Task::none() }
            Msg::CloseSettings => { self.view = View::Chat;     Task::none() }

            Msg::SetTemperature(v)   => { self.settings.temperature    = v; Task::none() }
            Msg::SetTopP(v)          => { self.settings.top_p          = v; Task::none() }
            Msg::SetMaxTokens(s)     => { self.settings.max_tokens     = s; Task::none() }
            Msg::ToggleResearch(v)   => { self.settings.enable_research = v; Task::none() }
            Msg::ToggleConfidence(v) => { self.settings.show_confidence = v; Task::none() }
            Msg::ToggleSeal(v) => {
                self.settings.enable_seal = v;
                if v && self.seal.is_none() && self.model.is_some() {
                    self.seal = Some(Arc::new(Mutex::new(SealLearner::new(SealConfig::default()))));
                }
                Task::none()
            }
        }
    }

    fn push(&mut self, role: Role, content: String, confidence: Option<f32>, sources: Vec<String>, elapsed_ms: u64) {
        self.messages.push(ChatMessage { role, content, confidence, sources, elapsed_ms });
    }

    fn run_command(&mut self, raw: &str) {
        let parts: Vec<&str> = raw.splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let arg = parts.get(1).copied().unwrap_or("").trim();
        match cmd.as_str() {
            "/help"     => self.push(Role::System, HELP.into(), None, vec![], 0),
            "/clear"    => { self.messages.clear(); self.history_ids.clear(); }
            "/research" => { self.settings.enable_research = arg == "on"; }
            "/learn"    => { self.settings.enable_seal    = arg == "on"; }
            "/local" if !arg.is_empty() => {
                let q = arg.to_string();
                let body = match KnowledgeStore::default().and_then(|ks| ks.search_local(&q)) {
                    Ok(e) if !e.is_empty() => e.iter().take(3)
                        .map(|e| format!("• {}: {}", e.query, e.summary))
                        .collect::<Vec<_>>().join("\n"),
                    _ => format!("Nothing found for '{}'.", q),
                };
                self.push(Role::System, body, None, vec![], 0);
            }
            _ => self.push(Role::System, "Unknown command. /help for list.".into(), None, vec![], 0),
        }
    }

    // ── Views ─────────────────────────────────────────────────────────────────

    pub fn view(&self) -> Element<'_, Msg> {
        match self.view {
            View::DatasetSelect => self.view_dataset_select(),
            View::Training      => self.view_training(),
            View::Chat          => self.view_chat(),
            View::Settings      => self.view_settings(),
        }
    }

    fn view_dataset_select(&self) -> Element<'_, Msg> {
        let dataset_items: Vec<Element<Msg>> = DATASETS.iter().enumerate().map(|(i, d)| {
            let sel = if self.selected_dataset == i { Some(i) } else { None };
            let _ = sel;
            column![
                radio(d.name, i, Some(self.selected_dataset), Msg::SelectDataset)
                    .size(14).text_size(13.0),
                Space::with_height(Length::Fixed(3.0)),
                container(text(d.desc).size(11.0).color(col::TEXT_MUTED))
                    .padding(Padding { left: 22.0, ..Default::default() }),
                Space::with_height(Length::Fixed(10.0)),
            ]
            .spacing(0)
            .into()
        }).collect();

        let custom = column![
            text("Or use a custom URL (HuggingFace .txt or .jsonl):").size(12.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(6.0)),
            text_input("https://huggingface.co/datasets/…", &self.custom_url)
                .on_input(Msg::SetCustomUrl)
                .size(12.0)
                .padding([7, 10])
                .style(rounded_input),
        ].spacing(0);

        let size_section = column![
            Space::with_height(Length::Fixed(24.0)),
            text("MODEL SIZE").size(11.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(10.0)),
            radio(ModelSize::Compact.label(), ModelSize::Compact, Some(self.selected_size), Msg::SelectSize)
                .size(14).text_size(12.0),
            Space::with_height(Length::Fixed(6.0)),
            radio(ModelSize::Small.label(), ModelSize::Small, Some(self.selected_size), Msg::SelectSize)
                .size(14).text_size(12.0),
            Space::with_height(Length::Fixed(6.0)),
            radio(ModelSize::Medium.label(), ModelSize::Medium, Some(self.selected_size), Msg::SelectSize)
                .size(14).text_size(12.0),
        ].spacing(0);

        let start_btn = button(
            text("Start Training").size(14.0).color(col::TEXT_LIGHT)
        )
        .on_press(Msg::StartTraining)
        .style(|_: &Theme, status| button::Style {
            background: Some(Background::Color(match status {
                button::Status::Hovered | button::Status::Pressed => col::ACCENT_HOVER,
                _ => col::ACCENT,
            })),
            border: border_r(8.0),
            text_color: col::TEXT_LIGHT,
            ..Default::default()
        })
        .padding([10, 24]);

        let card = column(dataset_items)
            .push(Space::with_height(Length::Fixed(6.0)))
            .push(custom)
            .push(size_section)
            .push(Space::with_height(Length::Fixed(32.0)))
            .push(start_btn);

        let header = column![
            text("Mimakiwa AI").size(26.0).color(col::TEXT),
            Space::with_height(Length::Fixed(6.0)),
            text("Train your own model from scratch — runs fully on your Mac.")
                .size(13.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(28.0)),
            text("TRAINING DATA").size(11.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(12.0)),
        ].spacing(0);

        let full = column![header]
            .push(card)
            .max_width(500);

        container(
            column![
                Space::with_height(Length::Fill),
                container(full).center_x(Length::Fill).padding(Padding::new(40.0)),
                Space::with_height(Length::Fill),
            ]
            .width(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_: &Theme| container::Style {
            background: Some(Background::Color(col::BG_MAIN)),
            ..Default::default()
        })
        .into()
    }

    fn view_training(&self) -> Element<'_, Msg> {
        let is_error  = self.loading_stage.starts_with("Error");
        let stage_col = if is_error { Color::from_rgb(0.80, 0.22, 0.22) } else { col::TEXT_MUTED };

        let step  = PROG_STEP.load(Relaxed);
        let total = PROG_TOTAL.load(Relaxed);
        let pct   = if total > 0 { step as f32 / total as f32 } else { 0.0 };

        let bar: Element<'_, Msg> = {
            let bar_w  = 360.0f32;
            let filled = (bar_w * pct).round();
            row![
                container(Space::with_width(Length::Fixed(filled)))
                    .height(Length::Fixed(3.0))
                    .style(|_: &Theme| container::Style {
                        background: Some(Background::Color(col::ACCENT)),
                        border: border_r(2.0),
                        ..Default::default()
                    }),
                container(Space::with_width(Length::Fixed(bar_w - filled)))
                    .height(Length::Fixed(3.0))
                    .style(|_: &Theme| container::Style {
                        background: Some(Background::Color(col::BORDER)),
                        border: border_r(2.0),
                        ..Default::default()
                    }),
            ]
            .into()
        };

        let note = if PROG_STAGE.load(Relaxed) == 3 {
            "Training on your Mac's GPU (Metal). First coherent output at ~500 steps."
        } else {
            "Setting up — this may take a moment…"
        };

        let card = column![
            text("Mimakiwa AI").size(28.0).color(col::TEXT),
            Space::with_height(Length::Fixed(6.0)),
            text("Training your model…").size(13.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(36.0)),
            bar,
            Space::with_height(Length::Fixed(14.0)),
            text(&self.loading_stage).size(12.0).color(stage_col),
            Space::with_height(Length::Fixed(16.0)),
            text(note).size(11.0).color(col::TEXT_MUTED),
        ]
        .spacing(0)
        .align_x(Alignment::Center)
        .max_width(400);

        container(
            column![
                Space::with_height(Length::Fill),
                container(card).center_x(Length::Fill),
                Space::with_height(Length::Fill),
            ]
            .width(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_: &Theme| container::Style {
            background: Some(Background::Color(col::BG_MAIN)),
            ..Default::default()
        })
        .into()
    }

    fn view_chat(&self) -> Element<'_, Msg> {
        if self.sidebar_open {
            row![self.sidebar(), self.chat_panel()]
                .height(Length::Fill)
                .into()
        } else {
            self.chat_panel()
        }
    }

    fn sidebar(&self) -> Element<'_, Msg> {
        let brand = container(text("Mimakiwa").size(15.0).color(col::TEXT))
            .padding(Padding { top: 18.0, right: 16.0, bottom: 10.0, left: 20.0 });

        let new_btn = button(
            row![
                text("+").size(14.0).color(col::TEXT),
                Space::with_width(Length::Fixed(7.0)),
                text("New chat").size(13.0).color(col::TEXT),
            ].align_y(Alignment::Center),
        )
        .on_press(Msg::NewChat)
        .style(|_: &Theme, s: button::Status| button::Style {
            background: Some(Background::Color(match s {
                button::Status::Hovered | button::Status::Pressed => col::BG_HOVER,
                _ => Color::TRANSPARENT,
            })),
            text_color: col::TEXT,
            border: border_r(6.0),
            shadow: Default::default(),
        })
        .width(Length::Fill)
        .padding([8, 12]);

        let sessions: Vec<Element<Msg>> = self.past_sessions.iter().enumerate().rev()
            .map(|(i, s)| {
                button(text(&s.title).size(12.0).color(col::TEXT_MUTED))
                    .on_press(Msg::RestoreSession(i))
                    .style(|_: &Theme, st: button::Status| button::Style {
                        background: Some(Background::Color(match st {
                            button::Status::Hovered | button::Status::Pressed => col::BG_HOVER,
                            _ => Color::TRANSPARENT,
                        })),
                        text_color: col::TEXT_MUTED,
                        border: border_r(5.0),
                        shadow: Default::default(),
                    })
                    .width(Length::Fill)
                    .padding([7, 12])
                    .into()
            })
            .collect();

        let status = if self.model.is_some() {
            text(format!("● {}", self.model_status)).size(11.0).color(col::GREEN)
        } else {
            text("○ loading").size(11.0).color(col::TEXT_MUTED)
        };

        let inner = column![brand, container(new_btn).padding(Padding::from([0u16, 8])), Space::with_height(Length::Fixed(4.0))]
            .extend(sessions)
            .push(Space::with_height(Length::Fill))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(container(status).padding(Padding::from([0u16, 20])))
            .push(Space::with_height(Length::Fixed(14.0)))
            .spacing(2);

        container(inner)
            .width(Length::Fixed(220.0))
            .height(Length::Fill)
            .style(|_: &Theme| container::Style {
                background: Some(Background::Color(col::BG_SIDEBAR)),
                border: Border { color: col::BORDER, width: 1.0, radius: 0.0.into() },
                ..Default::default()
            })
            .into()
    }

    fn chat_panel(&self) -> Element<'_, Msg> {
        let thinking: Element<'_, Msg> = if self.is_generating {
            text("Thinking…").size(12.0).color(col::TEXT_MUTED).into()
        } else {
            Space::with_width(Length::Shrink).into()
        };

        let topbar = container(
            row![
                button(panel_icon())
                    .on_press(Msg::ToggleSidebar)
                    .style(ghost_btn)
                    .padding([6, 8]),
                Space::with_width(Length::Fill),
                thinking,
                Space::with_width(Length::Fixed(10.0)),
                button(gear_icon())
                    .on_press(Msg::OpenSettings)
                    .style(ghost_btn)
                    .padding([6, 8]),
            ]
            .align_y(Alignment::Center)
            .spacing(4),
        )
        .padding(Padding { top: 10.0, right: 12.0, bottom: 10.0, left: 12.0 })
        .style(|_: &Theme| container::Style {
            background: Some(Background::Color(col::BG_SIDEBAR)),
            border: Border { color: col::BORDER, width: 0.0, radius: 0.0.into() },
            ..Default::default()
        })
        .width(Length::Fill);

        let bubbles: Vec<Element<Msg>> = self.messages.iter().map(bubble).collect();
        let msgs_col = column(bubbles).spacing(12).padding(Padding::from([16u16, 20]));

        let send_enabled = !self.input.trim().is_empty() && !self.is_generating;

        let send_btn = button(text("Send").size(13.0).color(col::TEXT_LIGHT))
            .style(|_: &Theme, status| button::Style {
                background: Some(Background::Color(match status {
                    button::Status::Hovered | button::Status::Pressed => col::ACCENT_HOVER,
                    _ => col::ACCENT,
                })),
                border: border_r(8.0),
                text_color: col::TEXT_LIGHT,
                ..Default::default()
            })
            .padding([9, 18]);

        let send_btn = if send_enabled {
            send_btn.on_press(Msg::Send)
        } else {
            send_btn
        };

        let input_row = container(
            row![
                text_input("Message Mimakiwa…", &self.input)
                    .on_input(Msg::TypeInput)
                    .on_submit(Msg::Send)
                    .size(14.0)
                    .padding([10, 12])
                    .style(rounded_input),
                Space::with_width(Length::Fixed(8.0)),
                send_btn,
            ]
            .align_y(Alignment::Center),
        )
        .padding(Padding { top: 10.0, right: 16.0, bottom: 14.0, left: 16.0 })
        .style(|_: &Theme| container::Style {
            background: Some(Background::Color(col::BG_MAIN)),
            border: Border { color: col::BORDER, width: 1.0, radius: 0.0.into() },
            ..Default::default()
        })
        .width(Length::Fill);

        column![
            topbar,
            scrollable(msgs_col)
                .id(scroll_id())
                .height(Length::Fill),
            input_row,
        ]
        .height(Length::Fill)
        .into()
    }

    fn view_settings(&self) -> Element<'_, Msg> {
        let back_btn = button(text("← Back").size(13.0).color(col::TEXT))
            .on_press(Msg::CloseSettings)
            .style(ghost_btn)
            .padding([6, 10]);

        let content = column![
            back_btn,
            Space::with_height(Length::Fixed(20.0)),
            text("SETTINGS").size(13.0).color(col::TEXT),
            Space::with_height(Length::Fixed(26.0)),
            text("GENERATION").size(11.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(14.0)),
            row![
                text("Temperature").size(13.0).color(col::TEXT).width(Length::Fixed(130.0)),
                slider(0.1f32..=2.0, self.settings.temperature, Msg::SetTemperature).step(0.05),
                Space::with_width(Length::Fixed(12.0)),
                text(format!("{:.2}", self.settings.temperature)).size(12.0).color(col::TEXT_MUTED)
                    .width(Length::Fixed(40.0)),
            ].align_y(Alignment::Center).spacing(10),
            Space::with_height(Length::Fixed(10.0)),
            row![
                text("Top-p").size(13.0).color(col::TEXT).width(Length::Fixed(130.0)),
                slider(0.1f32..=1.0, self.settings.top_p, Msg::SetTopP).step(0.05),
                Space::with_width(Length::Fixed(12.0)),
                text(format!("{:.2}", self.settings.top_p)).size(12.0).color(col::TEXT_MUTED)
                    .width(Length::Fixed(40.0)),
            ].align_y(Alignment::Center).spacing(10),
            Space::with_height(Length::Fixed(10.0)),
            row![
                text("Max tokens").size(13.0).color(col::TEXT).width(Length::Fixed(130.0)),
                text_input("200", &self.settings.max_tokens)
                    .on_input(Msg::SetMaxTokens).size(12.0).padding([6, 8])
                    .width(Length::Fixed(80.0))
                    .style(rounded_input),
            ].align_y(Alignment::Center).spacing(16),
            Space::with_height(Length::Fixed(30.0)),
            horizontal_rule(1),
            Space::with_height(Length::Fixed(26.0)),
            text("FEATURES").size(11.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(14.0)),
            checkbox("Web research — searches the web for factual questions", self.settings.enable_research)
                .on_toggle(Msg::ToggleResearch).size(15),
            Space::with_height(Length::Fixed(10.0)),
            checkbox("SEAL learning — records exchanges for fine-tuning", self.settings.enable_seal)
                .on_toggle(Msg::ToggleSeal).size(15),
            Space::with_height(Length::Fixed(10.0)),
            checkbox("Show confidence / timing", self.settings.show_confidence)
                .on_toggle(Msg::ToggleConfidence).size(15),
            Space::with_height(Length::Fixed(30.0)),
            horizontal_rule(1),
            Space::with_height(Length::Fixed(26.0)),
            text("MODEL").size(11.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(14.0)),
            text("Retrain from scratch with a different dataset or model size.")
                .size(12.0).color(col::TEXT_MUTED),
            Space::with_height(Length::Fixed(12.0)),
            button(text("Retrain model").size(13.0).color(col::TEXT_LIGHT))
                .on_press(Msg::RetrainModel)
                .style(|_: &Theme, status| button::Style {
                    background: Some(Background::Color(match status {
                        button::Status::Hovered | button::Status::Pressed => col::ACCENT_HOVER,
                        _ => col::ACCENT,
                    })),
                    border: border_r(8.0),
                    text_color: col::TEXT_LIGHT,
                    ..Default::default()
                })
                .padding([8, 20]),
        ]
        .spacing(0)
        .padding(Padding::new(40.0));

        container(scrollable(content).height(Length::Fill))
            .style(|_: &Theme| container::Style {
                background: Some(Background::Color(col::BG_MAIN)),
                ..Default::default()
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

// ── Bubble renderer ───────────────────────────────────────────────────────────

fn bubble(msg: &ChatMessage) -> Element<'_, Msg> {
    let is_user   = msg.role == Role::User;
    let is_system = msg.role == Role::System;

    if is_system {
        return container(text(&msg.content).size(12.0).color(col::TEXT_MUTED))
            .center_x(Length::Fill)
            .padding(Padding::from([4u16, 24]))
            .into();
    }

    let mut items: Vec<Element<Msg>> = vec![
        text(&msg.content)
            .size(14.0)
            .color(if is_user { col::TEXT_LIGHT } else { col::TEXT })
            .into(),
    ];

    if let Some(conf) = msg.confidence {
        let pct = (conf * 100.0).round() as u32;
        let elapsed = if msg.elapsed_ms >= 1000 {
            format!("{:.1}s", msg.elapsed_ms as f64 / 1000.0)
        } else {
            format!("{}ms", msg.elapsed_ms)
        };
        items.push(Space::with_height(Length::Fixed(4.0)).into());
        items.push(
            text(format!("{}% · {}", pct, elapsed))
                .size(11.0)
                .color(col::confidence_color(conf))
                .into()
        );
    }

    if !msg.sources.is_empty() {
        items.push(Space::with_height(Length::Fixed(8.0)).into());
        items.push(text("Sources").size(10.0).color(col::TEXT_MUTED).into());
        items.push(Space::with_height(Length::Fixed(3.0)).into());
        for src in msg.sources.iter().take(3) {
            let label = crate::nlp::domain_of(src);
            let url   = src.clone();
            items.push(
                button(text(label).size(11.0).color(col::ACCENT))
                    .on_press(Msg::OpenUrl(url))
                    .style(|_: &Theme, s: button::Status| button::Style {
                        background: Some(Background::Color(match s {
                            button::Status::Hovered | button::Status::Pressed =>
                                Color::from_rgba(0.325, 0.408, 0.471, 0.12),
                            _ => Color::TRANSPARENT,
                        })),
                        text_color: col::ACCENT,
                        border: border_r(4.0),
                        shadow: Default::default(),
                    })
                    .padding([2, 6])
                    .into()
            );
        }
    }

    container(column(items).spacing(2))
        .padding(Padding::from([10u16, 14]))
        .max_width(520)
        .style(move |_: &Theme| container::Style {
            background: Some(Background::Color(if is_user { col::BG_USER } else { col::BG_AI })),
            border: border_r(14.0),
            ..Default::default()
        })
        .into()
}

// ── Bootstrap (runs in background thread) ────────────────────────────────────

fn bootstrap(cfg: BootstrapConfig) -> BootstrapResult {
    use mimakiwa_train::{Trainer, TrainConfig};
    use mimakiwa_model::MimakiwaConfig;
    use mimakiwa_tokenizer::BPETokenizer;
    use std::io::Read as _;

    let dir = mimakiwa_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return BootstrapResult::Err(format!("mkdir: {e}"));
    }

    let model_path = dir.join("mimakiwa.bin");
    let tok_path   = dir.join("tokenizer.json");

    // 1. Download dataset
    PROG_STAGE.store(1, Relaxed);
    eprintln!("[bootstrap] downloading: {}", cfg.dataset_url);

    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(e) => return BootstrapResult::Err(format!("HTTP client: {e}")),
    };

    let mut resp = match client.get(&cfg.dataset_url).header("User-Agent", "mimakiwa/0.3").send() {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => return BootstrapResult::Err(format!("HTTP {}", r.status())),
        Err(e) => return BootstrapResult::Err(format!("Request: {e}")),
    };

    let max_bytes = cfg.dataset_max_mb * 1_048_576;
    let mut raw   = Vec::with_capacity(max_bytes.min(4 * 1_048_576));
    let mut buf   = [0u8; 65536];
    let mut downloaded = 0usize;

    loop {
        if downloaded >= max_bytes { break; }
        match resp.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let take = (max_bytes - downloaded).min(n);
                raw.extend_from_slice(&buf[..take]);
                downloaded += take;
                PROG_STEP.store(downloaded / 1024, Relaxed);
            }
        }
    }
    eprintln!("[bootstrap] downloaded {} MB", downloaded / 1_048_576);

    // Parse to plain text
    let corpus = parse_corpus(&raw, &cfg.dataset_format);
    if corpus.trim().is_empty() {
        return BootstrapResult::Err("Downloaded corpus is empty — check URL".into());
    }
    eprintln!("[bootstrap] corpus: {} chars", corpus.len());

    // 2. Build BPE tokenizer
    PROG_STAGE.store(2, Relaxed);
    eprintln!("[bootstrap] training BPE vocab={}", cfg.vocab_size);
    let mut tokenizer = BPETokenizer::new();
    tokenizer.train(&corpus, cfg.vocab_size);

    if let Err(e) = tokenizer.save(&tok_path) {
        return BootstrapResult::Err(format!("Tokenizer save: {e}"));
    }

    // 3. Build model
    let model_cfg = match cfg.model_size {
        ModelSize::Compact => MimakiwaConfig::compact(tokenizer.vocab_size()),
        ModelSize::Small   => MimakiwaConfig::small(tokenizer.vocab_size()),
        ModelSize::Medium  => MimakiwaConfig::medium(tokenizer.vocab_size()),
    };
    eprintln!("[bootstrap] model: embed={} layers={} heads={} params≈{:.1}M",
        model_cfg.embed_dim, model_cfg.n_layers, model_cfg.n_heads,
        model_cfg.param_count() as f32 / 1e6);

    let model = MimakiwaModel::new(model_cfg);

    // 4. Train
    PROG_STAGE.store(3, Relaxed);
    let train_cfg = TrainConfig {
        batch_size:     cfg.batch_size,
        seq_len:        cfg.seq_len,
        max_steps:      cfg.training_steps,
        warmup_steps:   (cfg.training_steps / 20).max(50),
        learning_rate:  cfg.learning_rate,
        checkpoint_dir: dir.clone(),
        save_interval:  9999999,
        ..TrainConfig::default()
    };

    PROG_TOTAL.store(cfg.training_steps, Relaxed);
    PROG_STEP.store(0, Relaxed);
    PROG_START.store(unix_now_ms(), Relaxed);

    eprintln!("[bootstrap] tokenising corpus…");
    let tokens = tokenizer.encode_fast(&corpus);
    eprintln!("[bootstrap] {} tokens, seq={}", tokens.len(), cfg.seq_len);

    if tokens.len() < cfg.seq_len + 1 {
        return BootstrapResult::Err("Not enough tokens for even one training sequence".into());
    }

    use mimakiwa_train::TextDataset;
    let dataset = TextDataset::from_tokens(tokens, cfg.seq_len);
    eprintln!("[bootstrap] dataset: {} sequences", dataset.len());

    let mut trainer = Trainer::new(model, train_cfg);
    for i in 0..cfg.training_steps {
        let batch = dataset.random_batch(cfg.batch_size);
        let stats = trainer.train_step_batch(&batch);
        PROG_STEP.store(i + 1, Relaxed);
        PROG_LOSS.store(stats.loss.to_bits(), Relaxed);
        PROG_TOKS.store(stats.tokens_per_sec.to_bits(), Relaxed);
        if i % 250 == 0 {
            eprintln!("[train] step {}/{} loss={:.4} lr={:.2e} tok/s={:.0}",
                i+1, cfg.training_steps, stats.loss, stats.lr, stats.tokens_per_sec);
        }
    }

    PROG_STEP.store(cfg.training_steps, Relaxed);

    // 5. Save
    PROG_STAGE.store(4, Relaxed);
    let params = trainer.model.n_params();
    let vocab  = tokenizer.vocab_size();

    if let Err(e) = trainer.model.save(&model_path) {
        return BootstrapResult::Err(format!("Model save: {e}"));
    }

    *LOADED.lock().unwrap() = Some(Loaded { model: trainer.model, tokenizer });
    BootstrapResult::Ok { params, vocab }
}

fn load_existing(
    model_path: &std::path::Path,
    tok_path:   &std::path::Path,
) -> BootstrapResult {
    PROG_STAGE.store(2, Relaxed);
    let tok = match BPETokenizer::load(tok_path) {
        Ok(t) => t,
        Err(e) => return BootstrapResult::Err(format!("Tokenizer: {e}")),
    };
    PROG_STAGE.store(3, Relaxed);
    let model = match MimakiwaModel::load(model_path) {
        Ok(m) => m,
        Err(e) => return BootstrapResult::Err(format!("Model: {e}")),
    };
    let params = model.n_params();
    let vocab  = tok.vocab_size();
    *LOADED.lock().unwrap() = Some(Loaded { model, tokenizer: tok });
    BootstrapResult::Ok { params, vocab }
}

// ── Corpus parsing ────────────────────────────────────────────────────────────

fn parse_corpus(raw: &[u8], format: &str) -> String {
    let text = String::from_utf8_lossy(raw);
    match format {
        "dolly_jsonl"      => parse_dolly_jsonl(&text),
        "openhermes_jsonl" => parse_openhermes_jsonl(&text),
        // Plain text (TinyStories, WikiText, etc.): wrap each paragraph/story in
        // User:/Assistant: format so the model learns to respond to prompts.
        _                  => wrap_plain_as_qa(&text),
    }
}

/// Wraps plain story/paragraph text in User:/Assistant: format.
/// The model must see this format during training to produce it at inference time.
fn wrap_plain_as_qa(text: &str) -> String {
    // Rotate through a small set of neutral prompts so the model learns to respond
    // to varied user inputs (not just one hardcoded phrase).
    const PROMPTS: &[&str] = &[
        "Tell me a story.",
        "Can you share a story?",
        "Write me a short story.",
        "Give me a story.",
        "Tell me something.",
    ];

    let mut out = String::with_capacity(text.len() + text.len() / 4);
    let mut prompt_idx = 0usize;

    // Split on blank lines — each chunk is one story / paragraph
    for chunk in text.split("\n\n") {
        let chunk = chunk.trim();
        if chunk.len() < 30 { continue; }  // skip noise / section headers
        let prompt = PROMPTS[prompt_idx % PROMPTS.len()];
        prompt_idx += 1;
        out.push_str("User: ");
        out.push_str(prompt);
        out.push_str("\nAssistant: ");
        out.push_str(chunk);
        out.push_str("\n\n");
    }
    out
}

fn parse_dolly_jsonl(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let inst = v["instruction"].as_str().unwrap_or("");
            let ctx  = v["context"].as_str().unwrap_or("");
            let resp = v["response"].as_str().unwrap_or("");
            if !resp.is_empty() {
                if ctx.is_empty() {
                    out.push_str(&format!("User: {inst}\nAssistant: {resp}\n\n"));
                } else {
                    out.push_str(&format!("User: {inst}\n{ctx}\nAssistant: {resp}\n\n"));
                }
            }
        }
    }
    out
}

fn parse_openhermes_jsonl(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(convs) = v["conversations"].as_array() {
                for pair in convs.windows(2) {
                    let from_h = pair[0]["from"].as_str().unwrap_or("") == "human";
                    let from_g = pair[1]["from"].as_str().unwrap_or("") == "gpt";
                    if from_h && from_g {
                        let q = pair[0]["value"].as_str().unwrap_or("");
                        let a = pair[1]["value"].as_str().unwrap_or("");
                        out.push_str(&format!("User: {q}\nAssistant: {a}\n\n"));
                    }
                }
            }
        }
    }
    out
}

// ── Generate (runs in background thread) ─────────────────────────────────────

fn generate(
    input:     String,
    hist_ids:  Vec<u32>,
    model:     Arc<Mutex<MimakiwaModel>>,
    tokenizer: Arc<BPETokenizer>,
    seal:      Option<Arc<Mutex<SealLearner>>>,
    settings:  Settings,
) -> GenResult {
    let gen_start = std::time::Instant::now();
    use crate::nlp::{self, Intent};

    let analysis = nlp::analyze(&input);

    if settings.enable_research && analysis.intent != Intent::Conversational {
        if let Some(result) = web_research(&analysis, &model, &tokenizer, &hist_ids, &settings, gen_start) {
            return result;
        }
    }

    let max_tok = settings.max_tokens.parse::<usize>().unwrap_or(200).clamp(32, 512);
    let max_ctx = 768usize.saturating_sub(max_tok);

    // Build prompt: trim history to fit context window
    let turn = tokenizer.encode_fast(&format!("\nUser: {}\nAssistant:", input.trim()));
    let hist_budget = max_ctx.saturating_sub(turn.len());
    let hist_slice: &[u32] = if hist_ids.len() > hist_budget {
        &hist_ids[hist_ids.len() - hist_budget..]
    } else {
        &hist_ids
    };

    let prompt: Vec<u32> = hist_slice.iter().chain(turn.iter()).copied().collect();

    let out = {
        let mg = model.lock().unwrap();
        mg.generate(&prompt, max_tok, settings.temperature, settings.top_p)
    };

    let reply_raw = tokenizer.decode(&out.tokens);
    let reply = truncate_at_stop(reply_raw.trim());
    let reply = if reply.is_empty() {
        "…".to_string()
    } else {
        reply
    };

    // SEAL: record this exchange
    if settings.enable_seal {
        if let Some(sa) = &seal {
            if let Ok(mut sg) = sa.try_lock() {
                sg.record_exchange(&input, &reply);
            }
        }
    }

    // Update history token buffer
    let input_ids  = tokenizer.encode_fast(&format!("\nUser: {}", input.trim()));
    let reply_ids  = tokenizer.encode_fast(&format!("\nAssistant: {}", reply));
    let mut new_hist = hist_ids;
    new_hist.extend_from_slice(&input_ids);
    new_hist.extend_from_slice(&reply_ids);
    if new_hist.len() > 2048 { new_hist.drain(0..new_hist.len() - 2048); }

    GenResult::Ok {
        text: reply,
        confidence: out.confidence,
        sources: vec![],
        history_ids: new_hist,
        elapsed_ms: gen_start.elapsed().as_millis() as u64,
    }
}

fn web_research(
    analysis:  &crate::nlp::QueryAnalysis,
    model:     &Arc<Mutex<MimakiwaModel>>,
    tokenizer: &Arc<BPETokenizer>,
    hist_ids:  &[u32],
    settings:  &Settings,
    gen_start: std::time::Instant,
) -> Option<GenResult> {
    use mimakiwa_search::{search_web, fetch_and_extract};
    use crate::nlp;

    let results = search_web(&analysis.search_query, 5).ok()?;
    if results.is_empty() { return None; }

    let mut pages = Vec::new();
    for r in results.iter().take(3) {
        if let Ok(page) = fetch_and_extract(&r.url) {
            if page.word_count > 40 { pages.push(page); }
        }
    }

    let mut all_paragraphs = Vec::new();
    let mut source_urls    = Vec::new();

    if pages.is_empty() {
        for r in results.iter().take(4) {
            let body = format!("{} {}", r.title, r.snippet);
            all_paragraphs.extend(nlp::extract_best_paragraphs(&body, &analysis.keywords, 1));
            source_urls.push(r.url.clone());
        }
    } else {
        for page in &pages {
            all_paragraphs.extend(nlp::extract_best_paragraphs(&page.text, &analysis.keywords, 2));
            source_urls.push(page.url.clone());
        }
    }

    let unique = nlp::deduplicate(all_paragraphs);
    let context: String = unique.iter()
        .flat_map(|p| p.split(". ").filter(|s| s.split_whitespace().count() > 4))
        .take(2)
        .map(|s| {
            let s = s.trim();
            if s.ends_with(['.', '?', '!']) { s.to_string() } else { format!("{s}.") }
        })
        .collect::<Vec<_>>()
        .join(" ");

    if context.trim().is_empty() { return None; }

    let max_tok = settings.max_tokens.parse::<usize>().unwrap_or(200).clamp(32, 256);
    let ctx_prompt = format!(
        "\nContext: {}\nUser: {}\nAssistant:",
        context, analysis.raw.trim()
    );
    let ctx_ids = tokenizer.encode_fast(&ctx_prompt);
    let hist_prefix = if hist_ids.len() > 256 { &hist_ids[hist_ids.len()-256..] } else { hist_ids };
    let prompt: Vec<u32> = hist_prefix.iter().chain(ctx_ids.iter()).copied().collect();

    let out = {
        let mg = model.lock().unwrap();
        mg.generate(&prompt, max_tok, settings.temperature.clamp(0.5, 0.8), settings.top_p)
    };

    let reply_raw = tokenizer.decode(&out.tokens);
    let answer = truncate_at_stop(reply_raw.trim());
    let answer = if answer.trim().is_empty() { context } else { answer };
    if answer.trim().is_empty() { return None; }

    save_research(&analysis.raw, &results, &pages, &answer);

    let input_ids  = tokenizer.encode_fast(&format!("\nUser: {}", analysis.raw.trim()));
    let reply_ids  = tokenizer.encode_fast(&format!("\nAssistant: {answer}"));
    let mut new_hist = hist_ids.to_vec();
    new_hist.extend_from_slice(&input_ids);
    new_hist.extend_from_slice(&reply_ids);
    if new_hist.len() > 2048 { new_hist.drain(0..new_hist.len() - 2048); }

    Some(GenResult::Ok {
        text: answer,
        confidence: if pages.is_empty() { 0.60 } else { 0.76 },
        sources: source_urls,
        history_ids: new_hist,
        elapsed_ms: gen_start.elapsed().as_millis() as u64,
    })
}

fn save_research(
    query:   &str,
    results: &[mimakiwa_search::SearchResult],
    pages:   &[mimakiwa_search::ScrapedPage],
    summary: &str,
) {
    if let Ok(ks) = KnowledgeStore::default() {
        let _ = ks.save_research(query, results, pages, summary);
    }
}

fn truncate_at_stop(s: &str) -> String {
    for stop in &["\nUser:", "\nAssistant:", "<|end", "###"] {
        if let Some(pos) = s.find(stop) {
            return s[..pos].trim().to_string();
        }
    }
    s.trim().to_string()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unix_now_ms() -> usize {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as usize
}

fn scroll_id() -> scrollable::Id { scrollable::Id::new("chat") }

fn gear_icon<'a>() -> Element<'a, Msg> {
    let handle = svg::Handle::from_memory(include_bytes!("icons/gear.svg").as_slice());
    svg(handle).width(18).height(18).into()
}

fn panel_icon<'a>() -> Element<'a, Msg> {
    let handle = svg::Handle::from_memory(include_bytes!("icons/panel.svg").as_slice());
    svg(handle).width(18).height(18).into()
}

fn border_r(r: f32) -> Border {
    Border { color: Color::TRANSPARENT, width: 0.0, radius: r.into() }
}

fn ghost_btn(_: &Theme, s: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(match s {
            button::Status::Hovered | button::Status::Pressed => col::BG_HOVER,
            _ => Color::TRANSPARENT,
        })),
        text_color: col::TEXT,
        border: border_r(6.0),
        shadow: Default::default(),
    }
}

fn rounded_input(_: &Theme, _: text_input::Status) -> text_input::Style {
    text_input::Style {
        background: Background::Color(col::BG_INPUT),
        border: Border { color: col::BORDER, width: 1.5, radius: 8.0.into() },
        icon: col::TEXT_MUTED,
        placeholder: col::TEXT_MUTED,
        value: col::TEXT,
        selection: Color::from_rgba(0.325, 0.408, 0.471, 0.35),
    }
}

const HELP: &str = "\
Commands:
  /help              — this message
  /clear             — clear chat + history
  /research on|off   — toggle web research
  /learn on|off      — toggle SEAL recording
  /local <query>     — search local knowledge";
