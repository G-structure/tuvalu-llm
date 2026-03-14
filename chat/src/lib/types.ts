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
