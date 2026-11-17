import { useEffect, useMemo, useState } from "react";
import { BookOpen, FileText, MessageCircle, RefreshCw, Sparkles, Trash2, Upload, WandSparkles, ClipboardCheck } from "lucide-react";
import { chat, deleteDocument, generateFlashcards, generateQuiz, generateSummary, listDocuments, scoreQuiz, uploadDocument } from "./api";

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [tab, setTab] = useState("chat");
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);
  const [flashcards, setFlashcards] = useState([]);
  const [cardCount, setCardCount] = useState(5);
  const [quiz, setQuiz] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizScore, setQuizScore] = useState(null);

  const selected = useMemo(() => documents.find((doc) => doc.id === selectedId), [documents, selectedId]);

  async function refresh() {
    try {
      const items = await listDocuments();
      setDocuments(items);
      if (!selectedId && items[0]) setSelectedId(items[0].id);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => { refresh(); }, []);

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true); setError("");
    try {
      const document = await uploadDocument(file);
      setDocuments((current) => [document, ...current]);
      setSelectedId(document.id); setMessages([]); setTab("chat");
    } catch (err) { setError(err.message); } finally { setUploading(false); }
  }

  async function handleDelete() {
    if (!selected || !window.confirm(`Delete ${selected.filename}?`)) return;
    try {
      await deleteDocument(selected.id);
      const remaining = documents.filter((doc) => doc.id !== selected.id);
      setDocuments(remaining); setSelectedId(remaining[0]?.id || null); setMessages([]);
    } catch (err) { setError(err.message); }
  }

  async function handleChat(event) {
    event.preventDefault();
    if (!question.trim() || !selected) return;
    const prompt = question.trim();
    const history = messages.slice(-6).map((item) => `${item.role.toUpperCase()}: ${item.content}`).join("\n");
    setQuestion(""); setBusy(true); setError("");
    setMessages((current) => [...current, { role: "user", content: prompt }, { role: "assistant", content: "" }]);
    try {
      const result = await chat(selected.id, prompt, history, (answer) => setMessages((current) => {
        const next = [...current]; next[next.length - 1] = { role: "assistant", content: answer }; return next;
      }));
      setMessages((current) => {
        const next = [...current];
        next[next.length - 1] = { role: "assistant", content: result.answer, citations: result.citations };
        return next;
      });
    } catch (err) {
      setError(err.message);
      setMessages((current) => {
        const next = [...current];
        if (next[next.length - 1]?.role === "assistant") {
          next[next.length - 1] = { role: "assistant", content: err.message };
        }
        return next;
      });
    } finally { setBusy(false); }
  }

  async function runSummary() {
    setBusy(true); setError("");
    try { setSummary(await generateSummary(selected.id)); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function runFlashcards() {
    setBusy(true); setError("");
    try { setFlashcards(await generateFlashcards(selected.id, cardCount)); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function runQuiz() {
    setBusy(true); setError(""); setQuizScore(null);
    try { setQuiz(await generateQuiz(selected.id, cardCount)); setQuizAnswers({}); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function submitQuiz(event) {
    event.preventDefault();
    setBusy(true); setError("");
    try { setQuizScore(await scoreQuiz(selected.id, quiz, quizAnswers)); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark">N</div><div><strong>NoteBot</strong><span>Grounded study workspace</span></div></div>
        <div className="top-actions"><span className="status-dot" /> API connected <button className="icon-button" onClick={refresh} title="Refresh"><RefreshCw size={17} /></button></div>
      </header>
      <main className="layout">
        <aside className="sidebar">
          <div className="sidebar-heading"><span>YOUR NOTES</span><label className="upload-button"><Upload size={15} /> Upload PDF<input type="file" accept="application/pdf" onChange={handleUpload} hidden /></label></div>
          {uploading && <div className="uploading">Processing document...</div>}
          <div className="document-list">
            {documents.map((doc) => <button key={doc.id} className={`document-item ${doc.id === selectedId ? "selected" : ""}`} onClick={() => { setSelectedId(doc.id); setMessages([]); }}><FileText size={17} /><span><b>{doc.filename}</b><small>{doc.page_count} pages · {doc.status}</small></span></button>)}
            {!documents.length && <div className="empty-sidebar">Upload a PDF to begin studying.</div>}
          </div>
          {selected && <button className="delete-button" onClick={handleDelete}><Trash2 size={15} /> Delete document</button>}
          <div className="sidebar-footer"><BookOpen size={16} /> Answers stay grounded in your notes.</div>
        </aside>
        <section className="workspace">
          {!selected ? <div className="welcome"><div className="welcome-icon"><Sparkles size={30} /></div><h1>Study smarter with your notes.</h1><p>Upload a PDF to chat with your material, create summaries, and generate flashcards.</p><label className="primary-button"><Upload size={17} /> Upload your first PDF<input type="file" accept="application/pdf" onChange={handleUpload} hidden /></label></div> :
            <><div className="document-header"><div><p className="eyebrow">ACTIVE DOCUMENT</p><h1>{selected.filename}</h1><span>{selected.page_count} pages · indexed and ready</span></div><div className="document-badge"><span className="status-dot" /> Ready</div></div>
              <nav className="tabs"><button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}><MessageCircle size={16} /> Chat</button><button className={tab === "summary" ? "active" : ""} onClick={() => setTab("summary")}><WandSparkles size={16} /> Summary</button><button className={tab === "flashcards" ? "active" : ""} onClick={() => setTab("flashcards")}><Sparkles size={16} /> Flashcards</button><button className={tab === "quiz" ? "active" : ""} onClick={() => setTab("quiz")}><ClipboardCheck size={16} /> Quiz</button></nav>
              {error && <div className="error">{error}</div>}
              {tab === "chat" && <div className="chat-panel"><div className="messages">{!messages.length && <div className="chat-empty"><div className="empty-orb"><MessageCircle size={25} /></div><p className="eyebrow">YOUR STUDY COPILOT</p><h2>What would you like to learn?</h2><p>Ask a question, clarify a concept, or use one of the suggestions below.</p><div className="prompt-grid"><button onClick={() => setQuestion("Give me a concise overview of these notes.")}>Give me an overview</button><button onClick={() => setQuestion("What are the most important concepts?")}>Key concepts</button><button onClick={() => setQuestion("Explain the hardest topic simply.")}>Explain simply</button></div></div>}{messages.map((message, index) => <div key={index} className={`message ${message.role}`}><div className="avatar">{message.role === "user" ? "Y" : "N"}</div><div className="message-card"><span className="message-label">{message.role === "user" ? "You" : "NoteBot"}</span><p>{message.content || "Thinking..."}</p>{message.citations?.length > 0 && <details className="sources"><summary>Sources <span>{message.citations.length}</span></summary><div className="source-list">{message.citations.map((citation) => <span key={`${citation.document_id}-${citation.page_number}`}>{citation.citation || `Page ${citation.page_number}`}</span>)}</div></details>}</div></div>)}</div><form className="composer" onSubmit={handleChat}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your notes..." disabled={busy} /><button disabled={busy || !question.trim()}><Sparkles size={17} /></button></form></div>}
              {tab === "summary" && <div className="tool-panel"><div className="tool-title"><WandSparkles size={22} /><div><h2>Document summary</h2><p>Turn your notes into a concise study guide.</p></div></div><button className="primary-button" onClick={runSummary} disabled={busy}>{busy ? "Analyzing..." : "Generate summary"}</button>{summary && <div className="result"><h2>{summary.title}</h2><p>{summary.summary}</p><h3>Key topics</h3><ul>{summary.key_topics.map((item) => <li key={item}>{item}</li>)}</ul><h3>Study tips</h3><ul>{summary.study_tips.map((item) => <li key={item}>{item}</li>)}</ul></div>}</div>}
              {tab === "flashcards" && <div className="tool-panel"><div className="tool-title"><Sparkles size={22} /><div><h2>Flashcards</h2><p>Practice the most important concepts in your notes.</p></div></div><div className="control-row"><label>Cards <select value={cardCount} onChange={(event) => setCardCount(Number(event.target.value))}>{[3, 5, 8, 10, 15].map((count) => <option key={count}>{count}</option>)}</select></label><button className="primary-button" onClick={runFlashcards} disabled={busy}>{busy ? "Creating..." : "Generate cards"}</button></div><div className="cards">{flashcards.map((card, index) => <article className="card" key={index}><span className={`difficulty ${card.difficulty}`}>{card.difficulty}</span><h3>{card.question}</h3><div className="answer">{card.answer}</div></article>)}</div></div>}
              {tab === "quiz" && <div className="tool-panel"><div className="tool-title"><ClipboardCheck size={22} /><div><h2>Quiz</h2><p>Check your understanding with multiple-choice and short-answer questions.</p></div></div>{!quiz && <button className="primary-button" onClick={runQuiz} disabled={busy}>{busy ? "Creating..." : "Generate quiz"}</button>}{quiz && <form onSubmit={submitQuiz}><div className="quiz-list">{quiz.questions.map((item, index) => <article className="result quiz-question" key={index}><span className={`difficulty ${item.difficulty}`}>{item.difficulty} · {item.question_type.replace("_", " ")}</span><h3>{index + 1}. {item.question}</h3>{item.question_type === "multiple_choice" ? item.options.map((option) => <label className="quiz-option" key={option}><input type="radio" name={`question-${index}`} value={option} checked={quizAnswers[index] === option} onChange={(event) => setQuizAnswers({ ...quizAnswers, [index]: event.target.value })} /> {option}</label>) : <input className="short-answer" value={quizAnswers[index] || ""} onChange={(event) => setQuizAnswers({ ...quizAnswers, [index]: event.target.value })} placeholder="Your answer" />}<small>Source pages: {item.source_pages.join(", ")}</small></article>)}</div><button className="primary-button" type="submit" disabled={busy}>{busy ? "Scoring..." : "Submit quiz"}</button>{quizScore && <div className="result"><h2>Score: {quizScore.score}/{quizScore.total}</h2><p>{quizScore.percentage}% correct</p></div>}</form>}</div>}
            </>}
        </section>
      </main>
    </div>
  );
}

export default App;
