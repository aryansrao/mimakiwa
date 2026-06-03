// Web search using DuckDuckGo HTML endpoint — no API key required
// Falls back to Bing HTML scraping if DDG fails

use anyhow::{Result, Context};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

// Primary search: DuckDuckGo HTML
pub fn search_ddg(query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
    let encoded = url_encode(query);
    let search_url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; Mimakiwa/0.1; research bot)")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let resp = client.get(&search_url)
        .header("Accept", "text/html")
        .send()
        .context("DDG request failed")?;

    let html = resp.text().context("DDG response not text")?;
    let doc = Html::parse_document(&html);

    let result_sel = Selector::parse(".result").unwrap();
    let title_sel  = Selector::parse(".result__title a").unwrap();
    let url_sel    = Selector::parse(".result__url").unwrap();
    let snippet_sel = Selector::parse(".result__snippet").unwrap();

    let mut results = Vec::new();
    for result in doc.select(&result_sel) {
        if results.len() >= max_results { break; }

        let title = result.select(&title_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let raw_url = result.select(&url_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let snippet = result.select(&snippet_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        if title.is_empty() || raw_url.is_empty() { continue; }

        // DDG sometimes returns relative URLs or adds tracking; normalize
        let url = normalize_url(&raw_url);
        if url.is_empty() { continue; }

        results.push(SearchResult { title, url, snippet });
    }

    Ok(results)
}

// Fallback: Bing HTML search
pub fn search_bing(query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
    let encoded = url_encode(query);
    let search_url = format!("https://www.bing.com/search?q={}&count={}", encoded, max_results);

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let resp = client.get(&search_url)
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .context("Bing request failed")?;

    let html = resp.text()?;
    let doc = Html::parse_document(&html);

    let result_sel  = Selector::parse("li.b_algo").unwrap();
    let title_sel   = Selector::parse("h2 a").unwrap();
    let snippet_sel = Selector::parse(".b_caption p").unwrap();
    // cite element holds the human-readable URL (e.g. "en.wikipedia.org › wiki › ...")
    let cite_sel    = Selector::parse("cite").unwrap();

    let mut results = Vec::new();
    for result in doc.select(&result_sel) {
        if results.len() >= max_results { break; }

        let title_elem = match result.select(&title_sel).next() {
            Some(e) => e,
            None    => continue,
        };

        let title = title_elem.text().collect::<String>().trim().to_string();
        if title.is_empty() { continue; }

        // Prefer the cite URL (real page URL) over Bing's click-tracking href
        let url = result.select(&cite_sel)
            .next()
            .map(|e| {
                let raw = e.text().collect::<String>().trim().to_string();
                // cite gives "en.wikipedia.org › wiki › Foo" — take first segment as domain+path
                let clean = raw.replace(" › ", "/");
                let clean = clean.split_whitespace().next().unwrap_or("").to_string();
                if clean.starts_with("http") { clean }
                else { format!("https://{}", clean) }
            })
            .unwrap_or_else(|| {
                // Fall back to href (may be a Bing redirect, but better than nothing)
                title_elem.value().attr("href").unwrap_or("").to_string()
            });

        if url.is_empty() || url.starts_with("javascript") { continue; }

        let snippet = result.select(&snippet_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        results.push(SearchResult { title, url, snippet });
    }

    Ok(results)
}

// Public API: tries DDG first, falls back to Bing
pub fn search_web(query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
    match search_ddg(query, max_results) {
        Ok(r) if !r.is_empty() => Ok(r),
        _ => {
            println!("  DDG returned no results, trying Bing...");
            search_bing(query, max_results)
        }
    }
}

fn url_encode(s: &str) -> String {
    s.chars().map(|c| match c {
        'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
        ' ' => '+'.to_string(),
        c => format!("%{:02X}", c as u32),
    }).collect()
}

fn normalize_url(raw: &str) -> String {
    let raw = raw.trim();
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return raw.to_string();
    }
    // Try to extract URL from DDG redirect (/l/?uddg=...)
    if let Some(pos) = raw.find("uddg=") {
        let encoded = &raw[pos + 5..];
        let decoded = encoded.split('&').next().unwrap_or("");
        let decoded = percent_decode(decoded);
        if decoded.starts_with("http") { return decoded; }
    }
    if raw.starts_with("//") {
        return format!("https:{}", raw);
    }
    String::new()
}

fn percent_decode(s: &str) -> String {
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i+1..i+3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    result.push(byte as char);
                    i += 3;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}
