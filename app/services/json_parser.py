#!/usr/bin/env python3
"""Centralized JSON parsing utilities for LLM responses."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """Custom exception for JSON parsing failures."""


class LLMJSONParser:
    """Robust JSON parser for LLM responses.

    Handles markdown cleanup, extraction, validation, and targeted repairs.
    """

    @staticmethod
    def _remove_trailing_commas(json_text: str) -> str:
        """Remove trailing commas before closing braces/brackets."""
        if not json_text:
            return json_text
        # ,  } or ,  ] → } or ]
        json_text = re.sub(r",\s*(\})", r"\1", json_text)
        json_text = re.sub(r",\s*(\])", r"\1", json_text)
        # Remove duplicate commas that can appear after line removals
        json_text = re.sub(r",\s*,", ",", json_text)
        return json_text

    @staticmethod
    def _dedupe_keys_in_object(object_text: str) -> str:
        """Remove duplicated keys in one object, keeping the last occurrence.

        Limited to expected fields to avoid over-aggressive removal.
        """
        if not object_text:
            return object_text

        pattern = (
            r'^(?P<indent>\s*)"'
            r'(?P<key>ticker|company|allocation|justification|confidence|risk|'
            r'sell_trigger)'
            r'"\s*:\s*'
        )
        key_pattern = re.compile(pattern, re.MULTILINE)

        lines = object_text.split("\n")
        seen_keys = set()
        # Walk from bottom to top to keep the last occurrence
        kept_reversed: list[str] = []
        for line in reversed(lines):
            m = key_pattern.match(line)
            if not m:
                kept_reversed.append(line)
                continue
            key = m.group("key")
            if key in seen_keys:
                # Skip earlier duplicate
                continue
            seen_keys.add(key)
            kept_reversed.append(line)

        repaired_lines = list(reversed(kept_reversed))
        repaired = "\n".join(repaired_lines)
        # Clean up trailing commas after possible line removals
        repaired = LLMJSONParser._remove_trailing_commas(repaired)
        return repaired

    @staticmethod
    def _repair_array_of_objects(json_text: str) -> str:
        """Attempt targeted repairs for a top-level JSON array of objects.

        - Deduplicate repeated keys inside each object (keep last)
        - Remove trailing commas
        """
        if not json_text or "[" not in json_text:
            return json_text

        # Extract first top-level array content (already handled earlier,
        # but safe)
        start = json_text.find("[")
        bracket_count = 0
        end = None
        for i, ch in enumerate(json_text[start:], start):
            if ch == "[":
                bracket_count += 1
            elif ch == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    end = i
                    break
        if end is None:
            return json_text

        prefix = json_text[:start]
        array_body = json_text[start + 1:end]
        suffix = json_text[end + 1:]

        # Split elements by balancing braces at top level
        elements: list[str] = []
        current = []
        brace_count = 0
        in_string = False
        escape = False
        for ch in array_body:
            current.append(ch)
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # End of an object; finalize element
                        element = "".join(current).strip()
                        elements.append(element)
                        current = []
        # Any trailing content is ignored for safety

        # Repair each object element
        repaired_elements: list[str] = []
        for el in elements:
            repaired_elements.append(LLMJSONParser._dedupe_keys_in_object(el))

        # Reassemble array with commas
        repaired_array = "[\n" + ",\n".join(repaired_elements) + "\n]"
        # Put back prefix/suffix if any, then remove trailing commas globally
        repaired_full = prefix + repaired_array + suffix
        repaired_full = LLMJSONParser._remove_trailing_commas(repaired_full)
        return repaired_full

    @staticmethod
    def _fix_unterminated_strings_robust(json_text: str) -> str:
        """More robust fix for unterminated strings by finding and closing them properly."""
        if not json_text:
            return json_text

        lines = json_text.splitlines()
        repaired_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this line starts a string value that's not properly closed
            if (('"justification":' in line or '"sell_trigger":' in line) and
                    not stripped.endswith('",')):
                # This is an unterminated string - collect all following lines
                string_lines = [line]
                i += 1

                # Collect all lines until we find a proper termination
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()

                    # If we find a closing quote or brace, we're done
                    if (next_stripped.endswith('",') or
                            next_stripped.endswith('"')):
                        string_lines.append(next_line)
                        break
                    elif (next_stripped.endswith('}') or
                          next_stripped.endswith('},')):
                        # End of object - close the string and add the brace
                        string_lines.append(next_line)
                        break
                    else:
                        string_lines.append(next_line)
                        i += 1

                # Now reconstruct the properly terminated string
                if string_lines:
                    first_line = string_lines[0]
                    key_part = first_line.split(':', 1)[0] + ': "'

                    # Extract all the content
                    content_parts = []
                    for sl in string_lines:
                        if sl == string_lines[0]:
                            # First line - get content after the colon
                            content = sl.split(':', 1)[1].strip().lstrip('"')
                        else:
                            content = sl.strip()
                        content_parts.append(content)

                    # Join all content and clean it up
                    full_content = ' '.join(content_parts)
                    # Remove any trailing braces or commas
                    full_content = full_content.rstrip('}, ')

                    # Create the properly terminated line
                    repaired_line = key_part + full_content + '",'
                    repaired_lines.append(repaired_line)

                    # If the last line was a closing brace, add it
                    last_line_stripped = string_lines[-1].strip()
                    if (last_line_stripped.endswith('}') or
                            last_line_stripped.endswith('},')):
                        repaired_lines.append(string_lines[-1])
            else:
                repaired_lines.append(line)

            i += 1

        return '\n'.join(repaired_lines)

    @staticmethod
    def _fix_unterminated_strings(json_text: str) -> str:
        """Aggressively fix unterminated strings by collapsing multi-line justifications."""
        lines = json_text.splitlines()
        repaired_lines = []
        in_multiline_justification = False
        justification_buffer = ""

        for line in lines:
            stripped = line.strip()

            if in_multiline_justification:
                # Add the content of the line to our buffer
                justification_buffer += " " + stripped.lstrip()

                # If this line terminates the object, we can close the justification
                if stripped.endswith("}") or stripped.endswith("},"):
                    # Clean up the buffer by removing trailing object closures or commas
                    justification_buffer = justification_buffer.rstrip("}, ")
                    # Add the closing quote and the comma
                    justification_buffer += '",'

                    # Find the last line that contains the justification key
                    last_justification_line_idx = -1
                    for j in range(len(repaired_lines) - 1, -1, -1):
                        if '"justification":' in repaired_lines[j]:
                            last_justification_line_idx = j
                            break

                    if last_justification_line_idx >= 0:
                        # Append the justification buffer to the last justification line
                        repaired_lines[last_justification_line_idx] += justification_buffer

                    # Add the rest of the line (e.g., the closing brace)
                    if "risk" in line:  # Find the next key to append
                        repaired_lines.append(line[line.find('"risk"'):])
                    else:  # If no other key, just close the object
                        repaired_lines.append("}")
                    in_multiline_justification = False
                    justification_buffer = ""
                continue

            if '"justification":' in line and not line.strip().endswith('",'):
                in_multiline_justification = True
                justification_buffer = line.split(':', 1)[1].strip().lstrip('"')
                # Keep the line up to the start of the string
                repaired_lines.append(line[:line.find(justification_buffer)])
            else:
                repaired_lines.append(line)

        return '\n'.join(repaired_lines)

    @staticmethod
    def _drop_unterminated_value_lines(json_text: str) -> str:
        """Remove lines where string values are clearly unterminated for
        known fields like justification/sell_trigger, keeping later valid
        duplicates intact."""
        if not json_text:
            return json_text
        keys = ("justification", "sell_trigger")
        out_lines: list[str] = []
        for line in json_text.split("\n"):
            stripped = line.strip()
            # If it's a target key line starting a string but lacks a closing quote
            is_target = any(
                stripped.startswith(f'"{k}"') for k in keys
            )
            if is_target and '"' in stripped:
                # After the first colon, count quotes to see if line closes
                # We consider closed if we see ..." or ...", at end
                after_colon = stripped.split(":", 1)[-1]
                closes_inline = bool(
                    re.search(r'"\s*,?\s*$', after_colon)
                )
                if not closes_inline:
                    # Drop this broken line; likely followed by a valid duplicate
                    continue
            out_lines.append(line)
        repaired = "\n".join(out_lines)
        # Clean any trailing commas created by removals
        return LLMJSONParser._remove_trailing_commas(repaired)

    @staticmethod
    def _add_missing_commas(json_text: str) -> str:
        """Add missing commas between properties in a JSON string."""
        # This regex looks for a property that ends (with a quote, number,
        # bool, or closing brace) and is immediately followed by the next
        # property's key on a new line, without a comma between them.
        repaired_text = re.sub(
            r'(["\d]|true|false|}|])\s*\n(\s*")', r'\1,\n\2', json_text
        )
        return repaired_text

    @staticmethod
    def clean_markdown_formatting(content: str) -> str:
        """Remove markdown code block formatting from content."""
        if not content or not isinstance(content, str):
            return content

        cleaned = content.strip()

        # Remove markdown code block patterns
        patterns = [
            (r"^```json\s*\n?", ""),
            (r"^```\s*\n?", ""),
            (r"\n?```\s*$", ""),
            (r"```\s*$", ""),
            (r"```json\s*", ""),
            (r"```\s*", ""),
        ]

        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE)

        return cleaned.strip()

    @staticmethod
    def extract_json_from_text(content: str) -> str:
        """Extract JSON array or object from mixed text content."""
        if not content:
            return content

        # Try to find JSON array first - match balanced brackets
        start = content.find("[")
        if start != -1:
            bracket_count = 0
            for i, char in enumerate(content[start:], start):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return content[start:i + 1]

        # Try to find JSON object - match balanced braces
        start = content.find("{")
        if start != -1:
            brace_count = 0
            for i, char in enumerate(content[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return content[start:i + 1]

        return content

    @staticmethod
    def validate_json_structure(
        data: Any,
        expected_type: Optional[str] = "list",
        _expected_length: Optional[int] = None,
    ) -> bool:
        """Validate that parsed JSON meets expected structure.

        Note: For lists, we validate the type and allow any length.
        Exact length handling (trim/accept) is performed at a higher
        level to avoid throwing away otherwise useful data.
        """
        if expected_type is None:
            # No type validation requested
            return True
        elif expected_type == "list":
            if not isinstance(data, list):
                return False
        elif expected_type == "dict":
            if not isinstance(data, dict):
                return False
        return True

    @classmethod
    def _parse_and_repair_json(
        cls,
        content: str,
        expected_type: Optional[str] = "list",
        expected_length: Optional[int] = None,
        fallback_value: Any = None,
    ) -> Any:
        """
        Parse JSON content from LLM response with robust error handling.

        Args:
            content: Raw content from LLM
            expected_type: Expected type ("list" or "dict")
            expected_length: Expected length for lists
            fallback_value: Value to return on parse failure

        Returns:
            Parsed JSON data or fallback_value

        Raises:
            JSONParseError: If parsing fails and no fallback provided
        """
        if not content:
            if fallback_value is not None:
                return fallback_value
            raise JSONParseError("Empty content provided")

        logger.debug("Original content length: %d", len(content))

        # Step 1: Clean markdown formatting
        cleaned_content = cls.clean_markdown_formatting(content)
        logger.debug("After markdown cleaning: %d chars", len(cleaned_content))

        # Step 2: Extract JSON if embedded in text (do this AFTER cleaning)
        if not cleaned_content.strip().startswith(("[", "{")):
            extracted_json = cls.extract_json_from_text(cleaned_content)
            if extracted_json != cleaned_content:
                cleaned_content = extracted_json
                logger.debug("Extracted JSON from mixed content")

        # Step 3: Final cleanup - ensure we only have the JSON part
        cleaned_content = cleaned_content.strip()

        # If first line looks like JSON, take only the complete JSON block
        if "\n" in cleaned_content:
            lines = cleaned_content.split("\n")
            if lines[0].strip().startswith(("[", "{")):
                # Try to find where the JSON ends
                if lines[0].strip().startswith("["):
                    bracket_count = 0
                    json_lines = []
                    for line in lines:
                        json_lines.append(line)
                        for char in line:
                            if char == "[":
                                bracket_count += 1
                            elif char == "]":
                                bracket_count -= 1
                        if bracket_count == 0:
                            cleaned_content = "\n".join(json_lines)
                            break
                elif lines[0].strip().startswith("{"):
                    brace_count = 0
                    json_lines = []
                    for line in lines:
                        json_lines.append(line)
                        for char in line:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                        if brace_count == 0:
                            cleaned_content = "\n".join(json_lines)
                            break

        # Step 4: Attempt JSON parsing
        try:
            parsed_data = json.loads(cleaned_content)
            logger.debug(
                "Successfully parsed JSON: %s",
                type(parsed_data),
            )

            # Step 5: Validate structure (type only; length handled by caller)
            if expected_type is not None and not cls.validate_json_structure(
                parsed_data, expected_type, expected_length
            ):
                error_msg = "Invalid structure: expected %s, got %s"
                logger.warning(
                    error_msg,
                    expected_type,
                    type(parsed_data),
                )
                if fallback_value is not None:
                    return fallback_value
                raise JSONParseError(
                    error_msg % (expected_type, type(parsed_data))
                )

            return parsed_data

        except json.JSONDecodeError as e:
            logger.error("Initial JSON decode failed: %s", e)
            
            # --- Start Layered Repair Sequence ---
            repaired_text = cleaned_content

            # Layer 1: Fix missing commas
            logger.debug("Applying Layer 1: Fix missing commas")
            repaired_text = cls._add_missing_commas(repaired_text)
            
            # Layer 2: Fix unterminated strings, a common Gemini issue
            logger.debug("Applying Layer 2: Fix unterminated strings")
            repaired_text = cls._fix_unterminated_strings_robust(repaired_text)
            
            # Layer 2b: Fallback to original method if needed
            logger.debug("Applying Layer 2b: Fix unterminated strings")
            repaired_text = cls._fix_unterminated_strings(repaired_text)

            # Layer 3: Drop lines with unterminated values
            logger.debug("Applying Layer 3: Drop unterminated lines")
            repaired_text = cls._drop_unterminated_value_lines(repaired_text)
            
            # Layer 4: Full repair for array of objects (dedupes keys)
            logger.debug("Applying Layer 4: Repair array")
            repaired_text = cls._repair_array_of_objects(repaired_text)
            
            # --- End Layered Repair Sequence ---
            
            try:
                parsed_data = json.loads(repaired_text)
                logger.warning(
                    "Parsed JSON successfully after layered repair sequence"
                )
                
                if not cls.validate_json_structure(
                    parsed_data, expected_type, expected_length
                ):
                    msg = "Invalid structure after full repair: expected %s, got %s"
                    logger.warning(msg, expected_type, type(parsed_data))
                    if fallback_value is not None:
                        return fallback_value
                    raise JSONParseError(msg % (expected_type, type(parsed_data)))
                
                return parsed_data
            
            except json.JSONDecodeError as e2:
                logger.error("All repair attempts failed: %s", e2)
                preview = repaired_text[:500].replace('\n', ' ')
                logger.error("Content after repair: %s...", preview)
                
                # Log the original content for debugging
                original_preview = cleaned_content[:500].replace('\n', ' ')
                logger.error("Original content: %s...", original_preview)

                if fallback_value is not None:
                    return fallback_value
                raise JSONParseError(f"All repair attempts failed: {e2}")

    @classmethod
    def parse_llm_json(
        cls,
        content: str,
        expected_type: str = "list",
        expected_length: Optional[int] = None,
        fallback_value: Any = None,
    ) -> Any:
        """
        Parse JSON content from LLM response with robust error handling.

        Args:
            content: Raw content from LLM
            expected_type: Expected type ("list" or "dict")
            expected_length: Expected length for lists
            fallback_value: Value to return on parse failure

        Returns:
            Parsed JSON data or fallback_value

        Raises:
            JSONParseError: If parsing fails and no fallback provided
        """
        if not content:
            if fallback_value is not None:
                return fallback_value
            raise JSONParseError("Empty content provided")

        logger.debug("Original content length: %d", len(content))

        # Step 1: Clean markdown formatting
        cleaned_content = cls.clean_markdown_formatting(content)
        logger.debug("After markdown cleaning: %d chars", len(cleaned_content))

        # Step 2: Extract JSON if embedded in text (do this AFTER cleaning)
        if not cleaned_content.strip().startswith(("[", "{")):
            extracted_json = cls.extract_json_from_text(cleaned_content)
            if extracted_json != cleaned_content:
                cleaned_content = extracted_json
                logger.debug("Extracted JSON from mixed content")

        # Step 3: Final cleanup - ensure we only have the JSON part
        cleaned_content = cleaned_content.strip()

        # If first line looks like JSON, take only the complete JSON block
        if "\n" in cleaned_content:
            lines = cleaned_content.split("\n")
            if lines[0].strip().startswith(("[", "{")):
                # Try to find where the JSON ends
                if lines[0].strip().startswith("["):
                    bracket_count = 0
                    json_lines = []
                    for line in lines:
                        json_lines.append(line)
                        for char in line:
                            if char == "[":
                                bracket_count += 1
                            elif char == "]":
                                bracket_count -= 1
                        if bracket_count == 0:
                            cleaned_content = "\n".join(json_lines)
                            break
                elif lines[0].strip().startswith("{"):
                    brace_count = 0
                    json_lines = []
                    for line in lines:
                        json_lines.append(line)
                        for char in line:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                        if brace_count == 0:
                            cleaned_content = "\n".join(json_lines)
                            break

        # Step 4: Attempt JSON parsing, with robust, layered repairs on failure
        try:
            return cls._parse_and_repair_json(
                cleaned_content, expected_type, expected_length, fallback_value
            )
        except JSONParseError as e:
            if fallback_value is not None:
                return fallback_value
            raise e

    @classmethod
    def parse_recommendation_list(
        cls, content: str, expected_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into recommendation list with specific
        validation.

        Behavior:
        - If more than expected_count items are returned, trim to
          expected_count and warn.
        - If fewer than expected_count items are returned but > 0,
          accept and warn.
        - If parsing fails entirely, return an empty list.
        - Handles both direct arrays and objects with "recommendations" key
        """
        try:
            # Parse as JSON array (expected format for all JSON mode models)
            recommendations = cls.parse_llm_json(
                content=content, expected_type="list", fallback_value=[]
            )

            if not isinstance(recommendations, list):
                logger.warning(f"Parsed data is not a list: {type(recommendations)}")
                return []

            # Length normalization
            if isinstance(recommendations, list):
                if expected_count is not None and len(recommendations) > \
                        expected_count:
                    logger.warning(
                        "Received %d items; trimming to %d",
                        len(recommendations),
                        expected_count,
                    )
                    recommendations = recommendations[:expected_count]
                elif (
                    expected_count is not None
                    and 0 < len(recommendations) < expected_count
                ):
                    logger.warning(
                        "Received %d items (< %d); accepting partial list",
                        len(recommendations),
                        expected_count,
                    )

            # Additional validation for recommendation structure
            if recommendations:
                required_fields = [
                    "ticker",
                    "company",
                    "allocation",
                    "justification",
                    "confidence",
                    "risk",
                ]
                for i, rec in enumerate(recommendations):
                    if not isinstance(rec, dict):
                        logger.warning(
                            "Recommendation %d is not a dict: %s",
                            i,
                            type(rec),
                        )
                        continue

                    missing_fields = [
                        field for field in required_fields if field not in rec
                    ]
                    if missing_fields:
                        logger.warning(
                            "Recommendation %d missing fields: %s",
                            i,
                            missing_fields,
                        )

            return recommendations

        except JSONParseError as e:
            logger.error(
                "Failed to parse recommendations: %s",
                e,
            )
            return []


class GeminiJSONConfig:
    """Configuration helper for Gemini JSON mode."""

    @staticmethod
    def get_recommendation_schema() -> Dict[str, Any]:
        """Get JSON schema for stock recommendations."""
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "company": {"type": "string"},
                    "allocation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "justification": {
                        "type": "string",
                        "maxLength": 200,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "sell_trigger": {"type": "string"},
                },
                "required": [
                    "ticker",
                    "company",
                    "allocation",
                    "justification",
                    "confidence",
                    "risk",
                    "sell_trigger",
                ],
            },
            "minItems": 10,
            "maxItems": 10,
        }

    @staticmethod
    def get_generation_config() -> Dict[str, Any]:
        """Get generation config for JSON mode."""
        return {
            "response_mime_type": "application/json",
            "response_schema": (
                GeminiJSONConfig.get_recommendation_schema()
            ),
            "temperature": 0.7,
            "max_output_tokens": 2000,
        }
