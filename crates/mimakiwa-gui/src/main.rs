mod app;
mod theme;
mod nlp;

fn main() -> iced::Result {
    iced::application("Mimakiwa AI", app::MimakiwaApp::update, app::MimakiwaApp::view)
        .theme(app::MimakiwaApp::theme)
        .subscription(app::MimakiwaApp::subscription)
        .run_with(app::MimakiwaApp::new)
}
