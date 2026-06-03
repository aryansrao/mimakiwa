// NLP pipeline: tokenize → intent classify → query build → TF-IDF extraction

use std::collections::HashMap;

// ── Word classes ──────────────────────────────────────────────────────────────

static STOPWORDS: &[&str] = &[
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","ought","just","also","only","even",
    "still","already","to","of","in","on","at","by","for","with","about",
    "against","between","through","during","before","after","above","below",
    "from","up","down","out","off","over","under","again","then","once",
    "and","but","or","nor","not","so","yet","both","either","neither",
    "that","this","these","those","i","me","my","we","our","you","your",
    "he","him","his","she","her","it","its","they","them","their",
    "all","each","every","few","more","most","other","some","such",
    "than","too","very","s","t","d","ll","re","ve","don","doesn","didn",
    "won","wouldn","couldn","let","tell","say","said","says","get","got",
    "go","going","goes","use","used","using","make","made","one","two",
    "three","many","much","new","good","well","back","way","come","want","see",
];

// Question words — used for intent detection but stripped from search keywords
static QUESTION_WORDS: &[&str] = &[
    "what","who","where","when","why","how","which","whose","whom",
    "whats","whos","wheres","whens","whys","hows","whose",
];

// Social/greeting words — indicate conversational input, not a searchable topic
static SOCIAL_WORDS: &[&str] = &[
    "hey","hi","hello","hiya","howdy","yo","sup","heya","greetings",
    "thanks","thank","cheers","appreciate","appreciated",
    "bye","goodbye","cya","later","ttyl","gtg",
    "ok","okay","alright","sure","cool","nice","great","awesome","wow",
    "yeah","yep","yup","nope","nah","no","yes",
    "lol","haha","hehe","lmao","omg","omfg","wtf",
    "dude","bro","mate","man","sis","bruh",
    "please","pls","plz","sorry","oops","my","bad",
];

fn is_stopword(w: &str) -> bool { STOPWORDS.contains(&w) }
fn is_question_word(w: &str) -> bool { QUESTION_WORDS.contains(&w) }
fn is_social_word(w: &str) -> bool { SOCIAL_WORDS.contains(&w) }

// ── Intent ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Definition,    // "what is X", "define X"
    HowTo,         // "how to", "how do I"
    Why,           // "why does", "why is"
    Factual,       // "who", "when", "where"
    Comparison,    // "X vs Y", "difference between"
    General,       // anything with content keywords
    Conversational, // social / no substantive keywords — use model generation
}

// ── Query analysis ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub raw: String,
    pub keywords: Vec<String>,
    pub search_query: String,
    pub intent: Intent,
}

pub fn analyze(input: &str) -> QueryAnalysis {
    let lower = input.trim().to_lowercase();

    let mut seen = std::collections::HashSet::new();

    // All non-stopword tokens (includes question words like "what", "where")
    let all_tokens: Vec<String> = lower.split_whitespace()
        .filter_map(|w| {
            let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
            if clean.len() >= 2 && !is_stopword(&clean) && seen.insert(clean.clone()) {
                Some(clean)
            } else {
                None
            }
        })
        .collect();

    // Substantive keywords: exclude question words AND social/greeting words
    // These are the words that carry real topic meaning
    let keywords: Vec<String> = all_tokens.iter()
        .filter(|w| !is_question_word(w) && !is_social_word(w))
        .cloned()
        .collect();

    // Classify intent using the full token set (needs question words for detection)
    let intent = classify_intent(&lower, &keywords, &all_tokens);
    let search_query = build_query(&intent, &keywords, input.trim());

    QueryAnalysis { raw: input.trim().to_string(), keywords, search_query, intent }
}

fn classify_intent(lower: &str, keywords: &[String], all_tokens: &[String]) -> Intent {
    // Conversational: no substantive topic words — route to model generation
    if keywords.is_empty() { return Intent::Conversational; }

    // Use all_tokens (which include question words) for pattern matching
    let has_token = |t: &str| all_tokens.iter().any(|w| w == t);

    if lower.starts_with("what is ") || lower.starts_with("what are ")
        || lower.starts_with("define ") || lower.starts_with("meaning of")
        || lower.starts_with("explain ")
        || (lower.contains("what") && lower.contains("mean"))
    {
        return Intent::Definition;
    }
    if lower.starts_with("how to ") || lower.starts_with("how do ")
        || lower.starts_with("how can ") || lower.starts_with("how does ")
        || lower.starts_with("how did ") || lower.starts_with("how ")
    {
        return Intent::HowTo;
    }
    if lower.starts_with("why ") || (has_token("why") && keywords.len() >= 1) {
        return Intent::Why;
    }
    if lower.starts_with("who ") || lower.starts_with("when ")
        || lower.starts_with("where ") || lower.starts_with("wheres ")
        || lower.starts_with("where's ")
    {
        return Intent::Factual;
    }
    if lower.contains(" vs ") || lower.contains(" versus ")
        || lower.contains("difference between") || lower.contains("compare ")
    {
        return Intent::Comparison;
    }
    Intent::General
}

fn build_query(intent: &Intent, keywords: &[String], raw: &str) -> String {
    if keywords.is_empty() { return raw.to_string(); }
    let kw = keywords.join(" ");
    match intent {
        Intent::Conversational => raw.to_string(),
        Intent::Definition     => format!("{} explained definition", kw),
        Intent::HowTo          => format!("how to {}", kw),
        Intent::Why            => format!("why {} reason", kw),
        Intent::Factual        => kw,
        Intent::Comparison     => format!("{} comparison", kw),
        Intent::General        => kw,
    }
}

// ── Paragraph-level TF-IDF extraction ────────────────────────────────────────

// Split text into paragraphs, score each by keyword relevance, return top N in doc order.
pub fn extract_best_paragraphs(text: &str, keywords: &[String], top_n: usize) -> Vec<String> {
    let paragraphs = split_paragraphs(text);
    if paragraphs.is_empty() { return vec![]; }
    if keywords.is_empty() {
        // No keywords — just return first few clean paragraphs
        return paragraphs.into_iter().take(top_n).collect();
    }

    let n = paragraphs.len() as f32;
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for p in &paragraphs {
        let unique: std::collections::HashSet<String> = tokenize_text(p).into_iter().collect();
        for w in unique { *doc_freq.entry(w).or_insert(0) += 1; }
    }

    let mut scored: Vec<(usize, f32, String)> = paragraphs.iter().enumerate().map(|(i, p)| {
        let words = tokenize_text(p);
        let wcount = words.len() as f32;
        if wcount < 8.0 { return (i, 0.0, p.clone()); }

        let word_freq: HashMap<&str, usize> =
            words.iter().fold(HashMap::new(), |mut m, w| {
                *m.entry(w.as_str()).or_insert(0) += 1; m
            });

        let mut score = 0.0f32;
        for kw in keywords {
            if let Some(&c) = word_freq.get(kw.as_str()) {
                let tf  = c as f32 / wcount;
                let df  = *doc_freq.get(kw).unwrap_or(&1) as f32;
                let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
                score  += tf * idf;
            }
        }
        // Slight bonus for paragraphs appearing early (intro/definition tend to be first)
        let pos_bonus = 1.0 / (1.0 + i as f32 * 0.08);
        (i, score * pos_bonus, p.clone())
    }).collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut top: Vec<(usize, f32, String)> = scored
        .into_iter()
        .filter(|(_, s, _)| *s > 0.0)
        .take(top_n)
        .collect();

    if top.is_empty() {
        // Nothing scored — return first few paragraphs by position
        return paragraphs.into_iter().take(top_n).collect();
    }

    top.sort_by_key(|(i, _, _)| *i);
    top.into_iter().map(|(_, _, p)| p).collect()
}

// Deduplicate paragraphs with >70% word overlap
pub fn deduplicate(paragraphs: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in paragraphs {
        let words: std::collections::HashSet<String> = tokenize_text(&p).into_iter().collect();
        let is_dup = out.iter().any(|existing| {
            let ew: std::collections::HashSet<String> = tokenize_text(existing).into_iter().collect();
            if words.is_empty() || ew.is_empty() { return false; }
            let overlap = words.intersection(&ew).count();
            let ratio = overlap as f32 / words.len().min(ew.len()) as f32;
            ratio > 0.70
        });
        if !is_dup { out.push(p); }
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn tokenize_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter_map(|w| {
            let c: String = w.chars().filter(|ch| ch.is_alphabetic()).collect();
            if c.len() >= 2 { Some(c) } else { None }
        })
        .collect()
}

fn split_paragraphs(text: &str) -> Vec<String> {
    // Try double-newline paragraphs first
    let raw_paras: Vec<&str> = text.split("\n\n").collect();
    let mut out = Vec::new();

    for para in raw_paras {
        let clean = para.trim();
        if clean.len() < 30 { continue; }
        if !is_clean_text(clean) { continue; }
        // If the paragraph is very long, split it at sentence boundaries
        if clean.split_whitespace().count() > 120 {
            for sent in split_sentences(clean) {
                if sent.split_whitespace().count() >= 8 && is_clean_text(&sent) {
                    out.push(sent);
                }
            }
        } else {
            out.push(clean.to_string());
        }
    }

    // If we didn't get good paragraphs, fall back to sentences
    if out.is_empty() {
        for sent in split_sentences(text) {
            if sent.split_whitespace().count() >= 8 && is_clean_text(&sent) {
                out.push(sent);
            }
        }
    }

    out
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') && cur.split_whitespace().count() >= 6 {
            let s = cur.trim().to_string();
            if !s.is_empty() { out.push(s); }
            cur = String::new();
        }
    }
    let tail = cur.trim().to_string();
    if tail.split_whitespace().count() >= 6 { out.push(tail); }
    out
}

fn is_clean_text(s: &str) -> bool {
    if s.starts_with("http") { return false; }
    if s.contains("&amp;") || s.contains("utm_") || s.contains("&p=") { return false; }
    if s.matches('=').count() > 3 || s.matches('&').count() > 2 { return false; }
    let alpha = s.chars().filter(|c| c.is_alphabetic() || c.is_whitespace()).count();
    alpha as f32 / s.len().max(1) as f32 > 0.68
}

pub fn domain_of(url: &str) -> String {
    let s = url.trim_start_matches("https://").trim_start_matches("http://");
    s.split('/').next().unwrap_or(url)
        .trim_start_matches("www.")
        .to_string()
}
