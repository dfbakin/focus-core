/**
 * Speech recognition for FocusCore (user scenario 6).
 *
 * The desktop application sends a short WAV clip of a critical moment
 * together with the signed-in user's token. The function checks the user,
 * forwards the clip to the OpenAI Whisper API with the key stored in the
 * project secrets and returns only the recognized text. The key never
 * leaves the server, and requests to OpenAI come from the function's region,
 * so users need neither their own key nor a VPN.
 */

const OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions";
const MODEL = "whisper-1";
const LANGUAGE = "ru";
const MAX_AUDIO_BYTES = 5 * 1024 * 1024;

function reply(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function isSignedIn(request: Request): Promise<boolean> {
  const authorization = request.headers.get("Authorization");
  const apikey = request.headers.get("apikey") ?? Deno.env.get("SUPABASE_ANON_KEY") ?? "";
  if (!authorization) {
    return false;
  }
  const response = await fetch(`${Deno.env.get("SUPABASE_URL")}/auth/v1/user`, {
    headers: { Authorization: authorization, apikey },
  });
  return response.ok;
}

Deno.serve(async (request: Request) => {
  if (request.method !== "POST") {
    return reply(405, { error_code: "method_not_allowed", msg: "Use POST" });
  }
  if (!(await isSignedIn(request))) {
    return reply(401, { error_code: "invalid_credentials", msg: "Sign in first" });
  }

  const form = await request.formData();
  const audio = form.get("file");
  if (!(audio instanceof File)) {
    return reply(400, { error_code: "no_audio", msg: "Send the audio in the file field" });
  }
  if (audio.size > MAX_AUDIO_BYTES) {
    return reply(413, { error_code: "audio_too_large", msg: "The clip is too long" });
  }

  const upstream = new FormData();
  upstream.append("file", audio, "moment.wav");
  upstream.append("model", MODEL);
  upstream.append("language", LANGUAGE);
  const response = await fetch(OPENAI_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${Deno.env.get("OPENAI_API_KEY")}` },
    body: upstream,
  });

  if (response.status === 429) {
    return reply(429, { error_code: "rate_limit", msg: "Transcription limit reached" });
  }
  if (!response.ok) {
    const details = await response.text();
    console.error(`OpenAI answered ${response.status}: ${details}`);
    return reply(502, {
      error_code: "transcription_failed",
      msg: `OpenAI answered ${response.status}: ${details.slice(0, 300)}`,
    });
  }
  const result = await response.json();
  return reply(200, { text: result.text ?? "" });
});
