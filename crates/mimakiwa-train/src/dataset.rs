// Text dataset for language model training
// Supports streaming from large files without loading everything into RAM

use mimakiwa_tokenizer::BPETokenizer;
use anyhow::Result;
use std::path::Path;

pub struct TextDataset {
    tokens: Vec<u32>,
    seq_len: usize,
    n_sequences: usize,
}

impl TextDataset {
    // Load text file, tokenize, and create dataset
    pub fn from_file(path: &Path, tokenizer: &BPETokenizer, seq_len: usize) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_text(&text, tokenizer, seq_len)
    }

    pub fn from_text(text: &str, tokenizer: &BPETokenizer, seq_len: usize) -> Result<Self> {
        println!("Tokenizing {} characters...", text.len());
        let tokens = tokenizer.encode_fast(text);
        println!("Got {} tokens", tokens.len());

        let n_sequences = tokens.len().saturating_sub(seq_len);
        Ok(Self { tokens, seq_len, n_sequences })
    }

    pub fn from_tokens(tokens: Vec<u32>, seq_len: usize) -> Self {
        let n_sequences = tokens.len().saturating_sub(seq_len);
        Self { tokens, seq_len, n_sequences }
    }

    pub fn len(&self) -> usize { self.n_sequences }
    pub fn is_empty(&self) -> bool { self.n_sequences == 0 }
    pub fn token_count(&self) -> usize { self.tokens.len() }

    // Get (input, target) pair at position i
    // input = tokens[i..i+seq_len], target = tokens[i+1..i+seq_len+1]
    pub fn get(&self, idx: usize) -> (&[u32], &[u32]) {
        let start = idx;
        let end = start + self.seq_len;
        (&self.tokens[start..end], &self.tokens[start + 1..end + 1])
    }

    // Random batch sampler — returns (inputs, targets) for a batch
    pub fn random_batch(&self, batch_size: usize) -> Vec<(&[u32], &[u32])> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..self.n_sequences);
                self.get(idx)
            })
            .collect()
    }

    /// Like random_batch but returns owned Vecs — enables rayon parallel construction.
    pub fn random_batch_owned(&self, batch_size: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        use rayon::prelude::*;
        use rand::Rng;
        let n = self.n_sequences;
        let sl = self.seq_len;
        // Generate all random indices first (can't be parallel due to RNG)
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..n.max(1)))
            .collect();
        // Parallel clone of token slices
        indices.par_iter()
            .map(|&idx| {
                let input  = self.tokens[idx..idx + sl].to_vec();
                let target = self.tokens[idx + 1..idx + sl + 1].to_vec();
                (input, target)
            })
            .collect()
    }

    // Iterator over sequential batches
    pub fn sequential_batches(&self, batch_size: usize) -> impl Iterator<Item = Vec<(&[u32], &[u32])>> {
        let mut start = 0;
        let n = self.n_sequences;
        let sl = self.seq_len;
        let tokens = &self.tokens;

        std::iter::from_fn(move || {
            if start >= n { return None; }
            let end = (start + batch_size).min(n);
            let batch: Vec<(&[u32], &[u32])> = (start..end)
                .map(|i| (&tokens[i..i + sl], &tokens[i + 1..i + sl + 1]))
                .collect();
            start += batch_size;
            Some(batch)
        })
    }

    // Shuffle the dataset by shuffling starting positions
    pub fn shuffled_indices(&self) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..self.n_sequences).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        indices
    }
}

// ---------------------------------------------------------------------------
// Format-aware training text parser
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum DataFormat {
    Plain,
    DollyJsonl,
    AlpacaJson,
    OpenHermesJsonl,
}

/// Parse raw downloaded text into training-ready plain text, given its format.
pub fn parse_training_text(raw: &str, format: &DataFormat) -> String {
    match format {
        DataFormat::Plain => raw.to_string(),
        DataFormat::DollyJsonl => parse_dolly_jsonl(raw),
        DataFormat::AlpacaJson => parse_alpaca_json(raw),
        DataFormat::OpenHermesJsonl => parse_openhermes_jsonl(raw),
    }
}

fn parse_dolly_jsonl(raw: &str) -> String {
    let mut out = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let instruction = v["instruction"].as_str().unwrap_or("").trim();
            let response    = v["response"].as_str().unwrap_or("").trim();
            let context     = v["context"].as_str().unwrap_or("").trim();
            if instruction.is_empty() || response.is_empty() { continue; }
            if context.is_empty() {
                out.push_str(&format!("User: {}\nAssistant: {}\n\n", instruction, response));
            } else {
                out.push_str(&format!("User: {}\n{}\nAssistant: {}\n\n", instruction, context, response));
            }
        }
    }
    out
}

fn parse_alpaca_json(raw: &str) -> String {
    let mut out = String::new();
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(raw) {
        for v in &arr {
            let instruction = v["instruction"].as_str().unwrap_or("").trim();
            let input       = v["input"].as_str().unwrap_or("").trim();
            let output      = v["output"].as_str().unwrap_or("").trim();
            if instruction.is_empty() || output.is_empty() { continue; }
            if input.is_empty() {
                out.push_str(&format!("User: {}\nAssistant: {}\n\n", instruction, output));
            } else {
                out.push_str(&format!("User: {}\n{}\nAssistant: {}\n\n", instruction, input, output));
            }
        }
    }
    out
}

fn parse_openhermes_jsonl(raw: &str) -> String {
    let mut out = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(convs) = v["conversations"].as_array() {
                let mut user_msg = "";
                let mut asst_msg = "";
                for c in convs {
                    match c["from"].as_str() {
                        Some("human") => user_msg = c["value"].as_str().unwrap_or(""),
                        Some("gpt") | Some("assistant") => asst_msg = c["value"].as_str().unwrap_or(""),
                        _ => {}
                    }
                }
                if !user_msg.is_empty() && !asst_msg.is_empty() {
                    out.push_str(&format!("User: {}\nAssistant: {}\n\n", user_msg.trim(), asst_msg.trim()));
                }
            }
        }
    }
    out
}

// Multi-file dataset for large corpora
pub struct MultiFileDataset {
    paths: Vec<std::path::PathBuf>,
    seq_len: usize,
    current_file: usize,
    current_dataset: Option<TextDataset>,
}

impl MultiFileDataset {
    pub fn new(paths: Vec<std::path::PathBuf>, seq_len: usize) -> Self {
        Self { paths, seq_len, current_file: 0, current_dataset: None }
    }

    pub fn load_next(&mut self, tokenizer: &BPETokenizer) -> Option<&TextDataset> {
        if self.current_file >= self.paths.len() { return None; }
        let path = &self.paths[self.current_file];
        println!("Loading file: {}", path.display());
        match TextDataset::from_file(path, tokenizer, self.seq_len) {
            Ok(ds) => {
                self.current_dataset = Some(ds);
                self.current_file += 1;
                self.current_dataset.as_ref()
            }
            Err(e) => {
                eprintln!("Error loading {}: {}", path.display(), e);
                self.current_file += 1;
                self.load_next(tokenizer)
            }
        }
    }
}
