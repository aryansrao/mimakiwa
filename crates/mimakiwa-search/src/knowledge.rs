// Local knowledge storage — organizes all research and conversations to disk
// Default location: ~/.mimakiwa/

use crate::search::SearchResult;
use crate::scraper::ScrapedPage;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub struct KnowledgeStore {
    pub base_dir: PathBuf,
}

// ─── Data types ──────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub date: String,
    pub turns: Vec<ConversationTurn>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,           // "user" or "assistant"
    pub content: String,
    pub sources: Vec<String>,   // URLs used (empty if not a research answer)
    pub confidence: f32,
    pub timestamp: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub query: String,
    pub summary: String,
    pub sources: Vec<String>,
    pub learned_at: String,   // ISO date string
    pub research_dir: String, // relative path under ~/.mimakiwa/research/
}

#[derive(Serialize, Deserialize)]
struct ResearchManifest {
    query: String,
    date: String,
    search_results: Vec<SearchResult>,
    scraped_pages: Vec<PageMeta>,
    summary_file: String,
}

#[derive(Serialize, Deserialize)]
struct PageMeta {
    url: String,
    title: String,
    word_count: usize,
    raw_file: String,
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl KnowledgeStore {
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(base_dir.join("sessions"))?;
        std::fs::create_dir_all(base_dir.join("research"))?;
        std::fs::create_dir_all(base_dir.join("knowledge"))?;
        Ok(Self { base_dir })
    }

    pub fn default() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self::new(PathBuf::from(home).join(".mimakiwa"))
    }

    // ── Research ──────────────────────────────────────────────────────────────

    /// Save full research results; returns the directory path created
    pub fn save_research(
        &self,
        query: &str,
        results: &[SearchResult],
        pages: &[ScrapedPage],
        summary: &str,
    ) -> Result<PathBuf> {
        let slug = slugify(query);
        let dir = self.base_dir.join("research").join(&slug);
        std::fs::create_dir_all(&dir)?;
        std::fs::create_dir_all(dir.join("raw"))?;

        // Save individual raw pages
        let mut page_metas = Vec::new();
        for page in pages {
            let hash = simple_hash(&page.url);
            let raw_file = format!("raw/{}.txt", hash);
            let content = format!("# {}\nURL: {}\n\n{}", page.title, page.url, page.text);
            std::fs::write(dir.join(&raw_file), &content)?;
            page_metas.push(PageMeta {
                url: page.url.clone(),
                title: page.title.clone(),
                word_count: page.word_count,
                raw_file,
            });
        }

        // Save summary as markdown
        let summary_content = format_summary_md(query, summary, results, pages);
        std::fs::write(dir.join("summary.md"), &summary_content)?;

        // Save manifest JSON
        let manifest = ResearchManifest {
            query: query.to_string(),
            date: current_date(),
            search_results: results.to_vec(),
            scraped_pages: page_metas,
            summary_file: "summary.md".to_string(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(dir.join("search_results.json"), manifest_json)?;

        Ok(dir)
    }

    /// Check if a similar query was already researched (keyword match)
    pub fn find_cached_research(&self, query: &str) -> Option<(PathBuf, String)> {
        let research_dir = self.base_dir.join("research");
        if !research_dir.exists() { return None; }

        let query_words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let entries = std::fs::read_dir(&research_dir).ok()?;
        let mut best_match: Option<(PathBuf, usize)> = None;

        for entry in entries.flatten() {
            let dir = entry.path();
            let manifest_path = dir.join("search_results.json");
            if !manifest_path.exists() { continue; }

            let manifest_str = std::fs::read_to_string(&manifest_path).ok()?;
            let manifest: ResearchManifest = serde_json::from_str(&manifest_str).ok()?;

            let match_count = query_words.iter()
                .filter(|&&w| manifest.query.to_lowercase().contains(&w.to_lowercase()))
                .count();

            if match_count > 0 && match_count >= query_words.len() / 2 {
                if best_match.as_ref().map_or(true, |(_, n)| match_count > *n) {
                    best_match = Some((dir, match_count));
                }
            }
        }

        if let Some((dir, _)) = best_match {
            let summary = std::fs::read_to_string(dir.join("summary.md")).ok()?;
            return Some((dir, summary));
        }

        None
    }

    // ── Conversations ─────────────────────────────────────────────────────────

    pub fn save_session(&self, session: &Session) -> Result<PathBuf> {
        let fname = format!("{}_{}.json", session.date, session.id);
        let path = self.base_dir.join("sessions").join(&fname);
        let json = serde_json::to_string_pretty(session)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    pub fn load_session(&self, session_id: &str) -> Result<Session> {
        let sessions_dir = self.base_dir.join("sessions");
        let entries = std::fs::read_dir(&sessions_dir)?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains(session_id) {
                let json = std::fs::read_to_string(entry.path())?;
                return serde_json::from_str(&json).context("Failed to parse session");
            }
        }
        Err(anyhow::anyhow!("Session not found: {}", session_id))
    }

    // ── Knowledge log ─────────────────────────────────────────────────────────

    pub fn append_knowledge(&self, entry: &KnowledgeEntry) -> Result<()> {
        let path = self.base_dir.join("knowledge").join("learned.jsonl");
        let line = serde_json::to_string(entry)? + "\n";
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }

    pub fn get_recent_knowledge(&self, n: usize) -> Result<Vec<KnowledgeEntry>> {
        let path = self.base_dir.join("knowledge").join("learned.jsonl");
        if !path.exists() { return Ok(Vec::new()); }
        let content = std::fs::read_to_string(&path)?;
        let entries: Vec<KnowledgeEntry> = content.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let start = entries.len().saturating_sub(n);
        Ok(entries[start..].to_vec())
    }

    /// Keyword search in local knowledge — returns matching entries
    pub fn search_local(&self, query: &str) -> Result<Vec<KnowledgeEntry>> {
        let all = self.get_recent_knowledge(500)?;
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, KnowledgeEntry)> = all.into_iter()
            .filter_map(|e| {
                let text = format!("{} {}", e.query, e.summary).to_lowercase();
                let score = words.iter().filter(|&&w| text.contains(w)).count();
                if score > 0 { Some((score, e)) } else { None }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(scored.into_iter().map(|(_, e)| e).take(5).collect())
    }

    pub fn base_dir(&self) -> &Path { &self.base_dir }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(60)
        .collect()
}

fn simple_hash(s: &str) -> String {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}

fn current_date() -> String {
    // Simple date without chrono dependency
    // Uses env or falls back to epoch placeholder
    std::env::var("DATE")
        .unwrap_or_else(|_| "2026-01-01".to_string())
}

fn format_summary_md(
    query: &str,
    summary: &str,
    results: &[SearchResult],
    pages: &[ScrapedPage],
) -> String {
    let mut md = format!("# Research: {}\n\n", query);
    md.push_str(&format!("**Date:** {}\n\n", current_date()));
    md.push_str("## Answer\n\n");
    md.push_str(summary);
    md.push_str("\n\n## Sources\n\n");
    for (i, r) in results.iter().enumerate() {
        md.push_str(&format!("{}. [{}]({})\n", i + 1, r.title, r.url));
        if !r.snippet.is_empty() {
            md.push_str(&format!("   > {}\n\n", r.snippet));
        }
    }
    md.push_str("\n## Scraped pages\n\n");
    for p in pages {
        md.push_str(&format!("- **{}** ({} words) — {}\n", p.title, p.word_count, p.url));
    }
    md
}
