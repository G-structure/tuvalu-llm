export interface Article {
  id: string;
  source_id: string;
  url: string;
  title_en: string;
  body_en: string;
  author: string | null;
  published_at: string;
  category: string | null;
  tags: string | null;
  image_url: string | null;
  image_alt: string | null;
  image_width: number | null;
  image_height: number | null;
  og_description_en: string | null;
  word_count: number | null;
  // From translations join
  title_tvl?: string | null;
  body_tvl?: string | null;
  og_description_tvl?: string | null;
}

export interface Category {
  slug: string;
  count: number;
}

export interface FeedbackSubmission {
  article_id: string;
  paragraph_idx: number;
  feedback_type: "thumbs_up" | "thumbs_down";
  island?: string;
  session_id?: string;
}

export interface SignalSubmission {
  article_id: string;
  signal_type: "share" | "reveal" | "thumbs_up" | "thumbs_down";
  paragraph_index?: number;
  session_id?: string;
  island?: string;
}

export interface IslandStats {
  island: string;
  count: number;
}

export interface FateleStats {
  total_this_month: number;
  islands: IslandStats[];
}

export const ISLANDS = [
  "Funafuti",
  "Vaitupu",
  "Nanumea",
  "Nui",
  "Nukufetau",
  "Niutao",
  "Nanumaga",
  "Nukulaelae",
  "Niulakita",
  "I fafo",
] as const;

export type Island = (typeof ISLANDS)[number];

// Chat types
export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponse {
  content: string;
  model_info?: {
    sampler_path: string;
    step: string;
  };
}
