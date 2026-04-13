use tempfile::NamedTempFile;

pub fn write_config(contents: &str) -> NamedTempFile {
    let file = NamedTempFile::new().expect("temp file should be created");
    std::fs::write(file.path(), contents).expect("config should be written");
    file
}
