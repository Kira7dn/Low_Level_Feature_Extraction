# Test Images for Unified Feature Extraction

This directory contains test images and their expected analysis results.

## Test Cases

| Name | Description | Preview | Text | Font | Text Color | BG Color |
|------|-------------|----------|------|------|------------|----------|
| simple_text | Test case: simple_text | <img src="simple_text.png" width="200" alt="simple_text"> | `Simple Test Text` | Arial 40pt | `#000000` | `#FFFFFF` |
| numbers | Test case: numbers | <img src="numbers.png" width="200" alt="numbers"> | `1234567890` | Arial 50pt | `#000000` | `#FFFFFF` |
| mixed_case | Test case: mixed_case | <img src="mixed_case.png" width="200" alt="mixed_case"> | `Mixed Case TeXt` | Arial 40pt | `#000000` | `#FFFFFF` |
| special_chars | Test case: special_chars | <img src="special_chars.png" width="200" alt="special_chars"> | `Special: !@#$%^&*()_+` | Arial 40pt | `#000000` | `#FFFFFF` |
| colored_text | Test case: colored_text | <img src="colored_text.png" width="200" alt="colored_text"> | `Colored Text` | Arial 45pt | `#460078` | `#F0F0FF` |
| dark_theme | Test case: dark_theme | <img src="dark_theme.png" width="200" alt="dark_theme"> | `Dark Theme` | Arial 42pt | `#DCDCDC` | `#222222` |

## Expected Results

Each test case has a corresponding JSON file with the expected analysis results. The results follow the analysis schema.
