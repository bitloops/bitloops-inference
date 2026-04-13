use serde_json::Value;

pub fn extract_json_object(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed)
        && value.is_object()
    {
        return Some(value);
    }

    let bytes = text.as_bytes();
    let mut candidate_start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaping = false;

    for (index, byte) in bytes.iter().enumerate() {
        if in_string {
            if escaping {
                escaping = false;
                continue;
            }

            match byte {
                b'\\' => escaping = true,
                b'"' => in_string = false,
                _ => {}
            }

            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    candidate_start = Some(index);
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0
                    && let Some(start) = candidate_start.take()
                {
                    let candidate = &text[start..=index];
                    if let Ok(value) = serde_json::from_str::<Value>(candidate)
                        && value.is_object()
                    {
                        return Some(value);
                    }
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_object_from_surrounding_text() {
        let value = extract_json_object(
            "Here is the result:\n{\"summary\":\"Adds a runtime\",\"confidence\":0.92}\nThanks.",
        )
        .expect("object should be extracted");

        assert_eq!(value["summary"], "Adds a runtime");
        assert_eq!(value["confidence"], 0.92);
    }

    #[test]
    fn ignores_invalid_json_fragments() {
        assert!(extract_json_object("No JSON object here").is_none());
    }
}
