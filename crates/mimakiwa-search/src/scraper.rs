// HTTP fetching + HTML content extraction
// Uses CSS selectors to strip nav/ads/footer and extract main text

use anyhow::{Result, Context};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScrapedPage {
    pub url: String,
    pub title: String,
    pub text: String,       // clean plaintext
    pub word_count: usize,
}

// Fetch URL and extract main content as plain text
pub fn fetch_and_extract(url: &str) -> Result<ScrapedPage> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; Mimakiwa/0.1; research bot)")
        .timeout(std::time::Duration::from_secs(12))
        .build()?;

    let resp = client.get(url)
        .header("Accept", "text/html,application/xhtml+xml")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .context(format!("Failed to fetch {}", url))?;

    // Only process text/html
    let content_type = resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !content_type.contains("text/html") && !content_type.contains("text/plain") {
        return Err(anyhow::anyhow!("Not HTML: {} for {}", content_type, url));
    }

    let html_str = resp.text().context("Failed to read response body")?;
    extract_content(url, &html_str)
}

pub fn extract_content(url: &str, html_str: &str) -> Result<ScrapedPage> {
    let doc = Html::parse_document(html_str);

    // Extract title
    let title = {
        let sel = Selector::parse("title").unwrap();
        doc.select(&sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default()
    };

    // Remove noise elements
    let noise_selectors = [
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", ".ad", ".ads", ".advertisement", ".cookie-notice",
        ".newsletter", ".sidebar", ".menu", "#navigation", "#footer",
        "#header", ".social-share", ".comments", "#comments",
    ];
    let doc = Html::parse_document(&remove_noise(html_str, &noise_selectors));

    // Priority content selectors (try in order, use first that gives substantial text)
    let content_selectors = [
        "article",
        "main",
        "[role='main']",
        ".content",
        ".post-content",
        ".article-body",
        ".entry-content",
        "#content",
        "#main-content",
        ".markdown-body",     // GitHub README
        ".mw-parser-output",  // Wikipedia
        "body",               // fallback
    ];

    let mut best_text = String::new();
    for sel_str in &content_selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            for elem in doc.select(&sel) {
                let text = elem.text()
                    .collect::<Vec<_>>()
                    .join(" ");
                let cleaned = clean_text(&text);
                if cleaned.len() > best_text.len() {
                    best_text = cleaned;
                }
                if best_text.split_whitespace().count() > 200 {
                    break;  // Good enough
                }
            }
            if best_text.split_whitespace().count() > 100 {
                break;
            }
        }
    }

    let word_count = best_text.split_whitespace().count();

    Ok(ScrapedPage {
        url: url.to_string(),
        title,
        text: best_text,
        word_count,
    })
}

// Split text into chunks that fit the model's context window
// max_chars: approximate character limit per chunk (~4 chars per token)
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in text.split('\n') {
        let para = paragraph.trim();
        if para.is_empty() { continue; }

        if current.len() + para.len() + 1 > max_chars {
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
            }
            // If single paragraph is too long, split at sentences
            if para.len() > max_chars {
                for sentence in split_sentences(para) {
                    if current.len() + sentence.len() > max_chars {
                        if !current.is_empty() {
                            chunks.push(current.trim().to_string());
                        }
                        current = sentence.to_string();
                    } else {
                        if !current.is_empty() { current.push(' '); }
                        current.push_str(&sentence);
                    }
                }
            } else {
                current = para.to_string();
            }
        } else {
            if !current.is_empty() { current.push('\n'); }
            current.push_str(para);
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    // Filter very short chunks
    chunks.into_iter().filter(|c| c.split_whitespace().count() > 10).collect()
}

fn clean_text(text: &str) -> String {
    // Collapse whitespace, remove excessive newlines
    let mut result = String::with_capacity(text.len());
    let mut prev_newline = false;
    let mut prev_space = false;

    for ch in text.chars() {
        match ch {
            '\n' | '\r' => {
                if !prev_newline {
                    result.push('\n');
                    prev_newline = true;
                }
                prev_space = false;
            }
            '\t' | ' ' => {
                if !prev_space && !prev_newline {
                    result.push(' ');
                    prev_space = true;
                }
            }
            c => {
                result.push(c);
                prev_newline = false;
                prev_space = false;
            }
        }
    }
    result.trim().to_string()
}

fn remove_noise(html: &str, noise_tags: &[&str]) -> String {
    // Simple regex-free tag removal for noise elements
    // For production, use scraper's element removal; this is a fast approximation
    let mut result = html.to_string();
    for tag in noise_tags {
        // Only remove by tag name (not class selectors) for simplicity
        if tag.starts_with('.') || tag.starts_with('#') || tag.starts_with('[') {
            continue;
        }
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);
        while let Some(start) = result.find(&open) {
            if let Some(end) = result[start..].find(&close) {
                let abs_end = start + end + close.len();
                result.replace_range(start..abs_end, "");
            } else {
                break;
            }
        }
    }
    result
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            if chars.peek().map_or(true, |&c| c == ' ' || c == '\n') {
                sentences.push(current.trim().to_string());
                current.clear();
            }
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}
