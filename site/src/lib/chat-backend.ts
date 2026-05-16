import type { APIEvent } from "@solidjs/start/server";

export const DEFAULT_CHAT_BACKEND_URL =
  "https://api.cyberneticphysics.com/tvl-chat";

export function getChatBackendUrl(event?: APIEvent): string {
  const cfEnv = (event?.context as any)?.cloudflare?.env;
  const configured =
    cfEnv?.CHAT_BACKEND_URL || process.env.CHAT_BACKEND_URL || DEFAULT_CHAT_BACKEND_URL;
  return String(configured).replace(/\/+$/, "");
}
