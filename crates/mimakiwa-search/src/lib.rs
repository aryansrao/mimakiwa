pub mod search;
pub mod scraper;
pub mod knowledge;
pub mod pipeline;

pub use search::{SearchResult, search_web};
pub use scraper::{ScrapedPage, fetch_and_extract, chunk_text};
pub use knowledge::{KnowledgeStore, KnowledgeEntry, Session, ConversationTurn};
