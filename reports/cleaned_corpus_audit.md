# Cleaned corpus audit
Input: `data/external/tv2en-cleaned/cleaned.jsonl`
Total rows: **176,157**

## Buckets by religious_density
Thresholds: low <= 0.02, high >= 0.08.

| bucket | rows | % |
|---|---:|---:|
| low | 58,949 | 33.5% |
| med | 85,117 | 48.3% |
| high | 32,091 | 18.2% |

## Bucket × domain
| domain | low | med | high | total |
|---|---:|---:|---:|---:|
| book | 39,467 | 69,443 | 28,875 | 137,785 |
| bible | 14,905 | 12,864 | 2,936 | 30,705 |
| dictionary | 4,308 | 7 | 95 | 4,410 |
| daily_text | 269 | 2,803 | 185 | 3,257 |

## Religious-token-count histogram (rows by count of JW-vocab tokens)
| count | rows |
|---:|---:|
| 0 | 45,672 |
| 1 | 19,079 |
| 2 | 28,722 |
| 3 | 13,924 |
| 4 | 11,841 |
| 5 | 7,133 |
| 6 | 5,997 |
| 7 | 4,772 |
| 8 | 4,435 |
| 9 | 3,763 |
| 10 | 3,525 |
| 11 | 3,058 |
| 12 | 2,942 |
| 13 | 2,684 |
| 14 | 2,382 |
| 15 | 2,203 |
| 16+ | 14,025 |
