// Byte-Pair Encoding (BPE) tokenizer — built from scratch
// 1. Initialize vocab with individual bytes (0-255)
// 2. Count adjacent pairs in training corpus
// 3. Merge most frequent pair iteratively until vocab_size is reached
// 4. Encode: apply merge rules; Decode: reverse mapping

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

const SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("<|pad|>", 0),
    ("<|bos|>", 1),
    ("<|eos|>", 2),
    ("<|unk|>", 3),
];

pub const PAD_TOKEN: u32 = 0;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
pub const UNK_TOKEN: u32 = 3;

#[derive(Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    // token_id -> token bytes
    pub vocab: Vec<Vec<u8>>,
    // token bytes -> token_id
    pub token_to_id: HashMap<Vec<u8>, u32>,
    // merge rules: (left_id, right_id) -> merged_id
    pub merges: Vec<(u32, u32, u32)>,
    // merge lookup for fast encoding
    merge_lookup: HashMap<(u32, u32), u32>,
    pub vocab_size: usize,
}

impl BPETokenizer {
    pub fn new() -> Self {
        let mut vocab = Vec::new();
        let mut token_to_id = HashMap::new();

        // Add special tokens first
        for (token_str, id) in SPECIAL_TOKENS {
            let bytes = token_str.as_bytes().to_vec();
            while vocab.len() <= *id as usize {
                vocab.push(Vec::new());
            }
            vocab[*id as usize] = bytes.clone();
            token_to_id.insert(bytes, *id);
        }

        // Add individual bytes (4..=259 covers 0..=255 after special tokens)
        for b in 0u8..=255 {
            let bytes = vec![b];
            let id = vocab.len() as u32;
            vocab.push(bytes.clone());
            token_to_id.insert(bytes, id);
        }

        let vocab_size = vocab.len();
        Self {
            vocab,
            token_to_id,
            merges: Vec::new(),
            merge_lookup: HashMap::new(),
            vocab_size,
        }
    }

    // Train BPE on text, expanding vocabulary to target_vocab_size
    pub fn train(&mut self, text: &str, target_vocab_size: usize) {
        // Tokenize text into byte-level token sequences, split at word boundaries
        let initial_ids = self.encode_bytes(text.as_bytes());

        // We work on a list of token sequences (one per "word" / chunk)
        let mut sequences = split_into_chunks(&initial_ids);

        println!("Training BPE: {} initial sequences, target vocab size {}", sequences.len(), target_vocab_size);

        let num_merges = target_vocab_size.saturating_sub(self.vocab.len());
        for merge_step in 0..num_merges {
            if sequences.is_empty() { break; }

            // Count all adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for seq in &sequences {
                for window in seq.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() { break; }

            // Find the most frequent pair
            let (&(left, right), &count) = pair_counts.iter()
                .max_by_key(|(_, &v)| v)
                .unwrap();

            if count < 2 { break; } // No pair appears more than once

            // Create new token
            let new_id = self.vocab.len() as u32;
            let mut new_bytes = self.vocab[left as usize].clone();
            new_bytes.extend_from_slice(&self.vocab[right as usize]);

            self.vocab.push(new_bytes.clone());
            self.token_to_id.insert(new_bytes, new_id);
            self.merges.push((left, right, new_id));
            self.merge_lookup.insert((left, right), new_id);

            // Apply merge to all sequences
            for seq in sequences.iter_mut() {
                apply_merge(seq, left, right, new_id);
            }

            if (merge_step + 1) % 100 == 0 {
                println!("  Merge {}/{}: ({}, {}) -> {} [freq={}]",
                    merge_step + 1, num_merges,
                    self.token_str(left), self.token_str(right),
                    new_id, count);
            }
        }

        self.vocab_size = self.vocab.len();
        println!("BPE training complete. Final vocab size: {}", self.vocab_size);
    }

    // Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode_bytes(text.as_bytes());
        // Apply all merge rules in order
        for &(left, right, merged) in &self.merges {
            apply_merge(&mut ids, left, right, merged);
        }
        ids
    }

    // Encode with BOS/EOS tokens
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![BOS_TOKEN];
        ids.extend(self.encode(text));
        ids.push(EOS_TOKEN);
        ids
    }

    // Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < 4 { continue; } // skip special tokens
            if id < self.vocab.len() as u32 {
                bytes.extend_from_slice(&self.vocab[id as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // Fast encode using merge lookup table
    pub fn encode_fast(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode_bytes(text.as_bytes());

        // Iteratively apply merges until no more can be applied
        loop {
            let mut changed = false;
            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() {
                    if let Some(&merged) = self.merge_lookup.get(&(ids[i], ids[i + 1])) {
                        new_ids.push(merged);
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
                new_ids.push(ids[i]);
                i += 1;
            }
            ids = new_ids;
            if !changed { break; }
        }
        ids
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        bytes.iter().map(|&b| {
            let bytes_token = vec![b];
            *self.token_to_id.get(&bytes_token).unwrap_or(&UNK_TOKEN)
        }).collect()
    }

    fn token_str(&self, id: u32) -> String {
        if id < self.vocab.len() as u32 {
            String::from_utf8_lossy(&self.vocab[id as usize]).into_owned()
        } else {
            format!("[{}]", id)
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        // Save as a JSON-friendly format (vocab as base64 strings)
        let save = SaveFormat {
            vocab: self.vocab.iter()
                .map(|bytes| bytes.iter().map(|b| *b as u32).collect())
                .collect(),
            merges: self.merges.clone(),
        };
        let json = serde_json::to_string(&save)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let save: SaveFormat = serde_json::from_str(&json)?;

        let vocab: Vec<Vec<u8>> = save.vocab.iter()
            .map(|ints| ints.iter().map(|&i| i as u8).collect())
            .collect();

        let token_to_id: HashMap<Vec<u8>, u32> = vocab.iter()
            .enumerate()
            .map(|(i, b)| (b.clone(), i as u32))
            .collect();

        let merge_lookup: HashMap<(u32, u32), u32> = save.merges.iter()
            .map(|&(l, r, m)| ((l, r), m))
            .collect();

        let vocab_size = vocab.len();
        Ok(Self {
            vocab,
            token_to_id,
            merges: save.merges,
            merge_lookup,
            vocab_size,
        })
    }
}

fn split_into_chunks(ids: &[u32]) -> Vec<Vec<u32>> {
    // Split at spaces (token for b' ' is 4 + b' ')
    let space_id = 4 + b' ' as u32; // byte 32 = space
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    for &id in ids {
        current.push(id);
        if id == space_id {
            if current.len() > 1 {
                chunks.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(ids.to_vec());
    }
    chunks
}

fn apply_merge(ids: &mut Vec<u32>, left: u32, right: u32, merged: u32) {
    let mut i = 0;
    let mut write = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
            ids[write] = merged;
            write += 1;
            i += 2;
        } else {
            ids[write] = ids[i];
            write += 1;
            i += 1;
        }
    }
    ids.truncate(write);
}

impl Default for BPETokenizer {
    fn default() -> Self { Self::new() }
}

#[derive(Serialize, Deserialize)]
struct SaveFormat {
    vocab: Vec<Vec<u32>>,
    merges: Vec<(u32, u32, u32)>,
}
