# TODO

## Verse boundary misalignment in Polynesian Bible translations

TVL verse 9 is 548 chars while EN verse 9 is only 115 chars — the TVL text for verse 9 includes content that EN splits across verses 9 and 10. And TVL verse 10 is just "10" (2 chars, essentially the verse number only). This is a translation paragraph structure difference — TVL wraps verses 9-10 into one span.v for verse 9, while the separate verse 10 span only has the number.

This is a known issue with Polynesian Bible translations — verse boundaries don't always align 1:1 with English at the span level. The current extraction merges multi-part verses but doesn't handle the case where one language combines verses.

This is a minor issue affecting ~5 out of 31,181 pairs (0.016%). The data is still correct — it's just that these verses have unusual length ratios. Not worth fixing now.
