const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const REFUSAL_MESSAGE = "I could not find enough information in your uploaded notes to answer this question.";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  return response;
}

export async function listDocuments() {
  return (await request("/documents")).json();
}

export async function uploadDocument(file) {
  const body = new FormData();
  body.append("file", file);
  return (await request("/documents", { method: "POST", body })).json();
}

export async function deleteDocument(id) {
  await request(`/documents/${id}`, { method: "DELETE" });
}

export async function chat(id, question, history, onToken) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);
  try {
    const response = await request(`/documents/${id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const token = decoder.decode(value, { stream: true });
      answer += token;
      onToken(answer);
    }
    return {
      answer,
      citations: answer.trim() === REFUSAL_MESSAGE
        ? []
        : JSON.parse(response.headers.get("X-Citations") || "[]"),
    };
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The AI service did not respond within 30 seconds. Please try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function generateSummary(id) {
  return (await request(`/documents/${id}/summary`, { method: "POST" })).json();
}

export async function generateFlashcards(id, count) {
  return (
    await request(`/documents/${id}/flashcards`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count }),
    })
  ).json();
}

export async function generateQuiz(id, count) {
  return (
    await request(`/documents/${id}/quiz`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count }),
    })
  ).json();
}

export async function scoreQuiz(id, quiz, answers) {
  return (
    await request(`/documents/${id}/quiz/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quiz, answers }),
    })
  ).json();
}
