import { useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, FileText, Menu, MessageCircle, RefreshCw, Sparkles, Trash2, Upload, WandSparkles, ClipboardCheck, Eye, X, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { createConversation, deleteConversation, deleteDocument, documentFileUrl, generateFlashcards, generateQuiz, generateSummary, getConversation, listAssets, listConversations, listDocuments, listVersions, scoreQuiz, sendMessage, uploadDocument } from "./api";

function App() {
  const [documents, setDocuments] = useState([]);
  const [versions, setVersions] = useState({});
  const [assets, setAssets] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [pinnedVersion, setPinnedVersion] = useState(null);
  const [newVersionNotice, setNewVersionNotice] = useState(null);
  const [question, setQuestion] = useState("");
  const [tab, setTab] = useState("chat");
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);
  const [flashcards, setFlashcards] = useState([]);
  const [savedAssetId, setSavedAssetId] = useState(null);
  const [pdfOpen, setPdfOpen] = useState(false);
  const [pdfError, setPdfError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [cardCount, setCardCount] = useState(5);
  const [quiz, setQuiz] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizScore, setQuizScore] = useState(null);
  const skipConversationHydration = useRef(false);

  const selected = useMemo(() => {
    const direct = documents.find((doc) => doc.id === selectedId);
    if (direct) return direct;
    const version = Object.values(versions).flat().find((item) => item.id === selectedId);
    if (version) return version;
    const active = conversations.find((item) => item.id === conversationId);
    return documents.find((doc) => doc.document_id === active?.document_id);
  }, [documents, selectedId, conversations, conversationId]);

  const latestVersion = selected
    ? (versions[selected.document_id] || []).find((version) => version.is_current)
      || versions[selected.document_id]?.[0]
      || selected
    : null;
  const pinnedVersionIsOutdated = Boolean(
    pinnedVersion
    && latestVersion
    && latestVersion.version_number > pinnedVersion.version_number
  );
  const selectedAssets = selected
    ? assets.filter((asset) => asset.document_id === selected.document_id)
    : [];
  const assetHistory = (assetType) => selectedAssets
    .filter((asset) => asset.asset_type === assetType);

  function openSavedAsset(asset) {
    setSavedAssetId(asset.id);
    if (asset.asset_type === "summary") {
      setSummary(asset.payload);
      setTab("summary");
    } else if (asset.asset_type === "flashcards") {
      setFlashcards(asset.payload.flashcards || asset.payload.cards || asset.payload);
      setTab("flashcards");
    } else if (asset.asset_type === "quiz") {
      setQuiz(asset.payload);
      setQuizAnswers({});
      setQuizScore(null);
      setTab("quiz");
    }
  }

  async function refresh() {
    try {
      const items = await listDocuments();
      const storedConversations = await listConversations();
      setDocuments(items);
      const versionLists = await Promise.all(
        items.map(async (item) => [item.document_id, await listVersions(item.document_id)])
      );
      setVersions(Object.fromEntries(versionLists));
      const assetLists = await Promise.all(
        items.map(async (item) => listAssets(item.document_id))
      );
      setAssets(assetLists.flat());
      setConversations(storedConversations);
      const savedConversationId = window.localStorage.getItem("notebot.conversationId");
      const restored = storedConversations.find((item) => item.id === savedConversationId)
        || storedConversations[0];
      if (restored) {
        setConversationId(restored.id);
        setSelectedId(restored.document_version_id);
        setPinnedVersion(restored);
      } else if (items[0]) {
        setSelectedId(items[0].id);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    if (!conversationId) return;
    window.localStorage.setItem("notebot.conversationId", conversationId);
    if (skipConversationHydration.current) {
      skipConversationHydration.current = false;
      return;
    }
    getConversation(conversationId)
      .then((conversation) => {
      setMessages((conversation.messages || []).map((message) => ({
        ...message,
        citations: message.citations || message.sources || [],
      })));
        setPinnedVersion(conversation);
        setSelectedId(conversation.document_version_id);
      })
      .catch((err) => setError(err.message));
  }, [conversationId]);

  useEffect(() => {
    window.localStorage.setItem("notebot.tab", tab);
  }, [tab]);

  useEffect(() => {
    setPdfOpen(false);
    setPdfError("");
  }, [selectedId]);

  async function openPdf() {
    if (!selected) return;
    setPdfError("");
    try {
      const response = await fetch(documentFileUrl(selected.id));
      const contentType = response.headers.get("content-type") || "";
      if (!response.ok || !contentType.includes("application/pdf")) {
        setPdfError(
          "This document version was uploaded before PDF preview storage was enabled. Re-upload the same PDF to enable preview."
        );
        setPdfOpen(true);
        return;
      }

      setPdfOpen(true);
      setTab("chat");
    } catch (err) {
      setPdfError(`The PDF preview could not be opened: ${err.message}`);
      setPdfOpen(true);
    }
  }

  function togglePdf() {
    if (pdfOpen) {
      setPdfOpen(false);
      return;
    }
    openPdf();
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true); setError("");
    try {
      const document = await uploadDocument(file);
      if (document.version_number > 1) setNewVersionNotice(document);
      setDocuments((current) => [document, ...current]);
      setSelectedId(document.id); setMessages([]); setConversationId(null); setPinnedVersion(null); setTab("chat");
      await refresh();
    } catch (err) { setError(err.message); } finally { setUploading(false); }
  }

  async function startNewChatWithLatest() {
    if (!latestVersion) return;
    setBusy(true); setError("");
    try {
      const conversation = await createConversation(
        latestVersion.id,
        `New chat - ${latestVersion.name || latestVersion.filename} v${latestVersion.version_number}`,
      );
      setConversations((current) => [conversation, ...current]);
      setConversationId(conversation.id);
      setPinnedVersion(conversation);
      setSelectedId(latestVersion.id);
      setMessages([]);
      setNewVersionNotice(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    if (!selected || !window.confirm(`Delete ${selected.filename}?`)) return;
    try {
      await deleteDocument(selected.id);
      const remaining = documents.filter((doc) => doc.id !== selected.id);
      setDocuments(remaining); setSelectedId(remaining[0]?.id || null); setMessages([]); setConversationId(null); setPinnedVersion(null);
      window.localStorage.removeItem("notebot.conversationId");
      await refresh();
    } catch (err) { setError(err.message); }
  }

  async function handleChat(event) {
    event.preventDefault();
    if (!question.trim() || !selected) return;
    const prompt = question.trim();
    setQuestion(""); setBusy(true); setError("");
    setMessages((current) => [...current, { role: "user", content: prompt }, { role: "assistant", content: "" }]);
    try {
      let activeConversationId = conversationId;
      if (!activeConversationId) {
        const conversation = await createConversation(selected.id);
        activeConversationId = conversation.id;
        skipConversationHydration.current = true;
        setConversationId(activeConversationId);
        setPinnedVersion(conversation);
        setConversations((current) => [conversation, ...current]);
      }
      const result = await sendMessage(activeConversationId, prompt, (answer) => setMessages((current) => {
        const next = [...current]; next[next.length - 1] = { role: "assistant", content: answer }; return next;
      }));
      setConversationId(result.conversationId || activeConversationId);
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
    try { setSummary(await generateSummary(selected.id)); await refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function runFlashcards() {
    setBusy(true); setError("");
    try { setFlashcards(await generateFlashcards(selected.id, cardCount)); await refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function runQuiz() {
    setBusy(true); setError(""); setQuizScore(null);
    try { setQuiz(await generateQuiz(selected.id, cardCount)); setQuizAnswers({}); await refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); }
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
        <div className="top-actions"><span className="status-dot" /> API connected <button className="icon-button panel-toggle" onClick={() => setSidebarOpen((open) => !open)} title={sidebarOpen ? "Close notes sidebar" : "Open notes sidebar"} aria-label={sidebarOpen ? "Close notes sidebar" : "Open notes sidebar"}><span className="panel-toggle-icon">{sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}</span><span className="panel-toggle-label">Notes</span></button><button className="icon-button panel-toggle" onClick={togglePdf} title={pdfOpen ? "Close PDF panel" : "Open PDF panel"} aria-label={pdfOpen ? "Close PDF panel" : "Open PDF panel"}><span className="panel-toggle-icon">{pdfOpen ? <X size={19} /> : <Menu size={19} />}</span><span className="panel-toggle-label">PDF</span></button><button className="icon-button" onClick={refresh} title="Refresh" aria-label="Refresh"><RefreshCw size={17} /></button></div>
      </header>
      <main className="layout">
        <aside className={`sidebar ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
          <div className="sidebar-heading"><span>YOUR NOTES</span><label className="upload-button"><Upload size={15} /> Upload PDF<input type="file" accept="application/pdf" onChange={handleUpload} hidden /></label></div>
          {uploading && <div className="uploading">Processing document...</div>}
          <div className="document-list">
            {documents.filter((doc) => doc.is_current !== false).map((doc) => <div key={doc.document_id}><button className={`document-item ${doc.id === selectedId ? "selected" : ""}`} onClick={() => { setSelectedId(doc.id); setMessages([]); setConversationId(null); setPinnedVersion(null); }}><FileText size={17} /><span><b>{doc.name || doc.filename}</b><small>Current · {doc.page_count} pages</small></span></button><div className="version-list">{(versions[doc.document_id] || [doc]).map((version) => <button key={version.id} className={`version-item ${version.id === selectedId ? "selected" : ""}`} onClick={() => { setSelectedId(version.id); setMessages([]); setConversationId(null); setPinnedVersion(null); }}>Version {version.version_number}{version.is_current ? " · Current" : ""}</button>)}</div></div>)}
            {!documents.length && <div className="empty-sidebar">Upload a PDF to begin studying.</div>}
          </div>
          <div className="conversation-sidebar">
            <div className="sidebar-section-label">CONVERSATIONS</div>
            {!conversations.length && <div className="empty-sidebar">No saved conversations yet.</div>}
            {conversations.map((conversation) => <button key={conversation.id} className={`conversation-item ${conversation.id === conversationId ? "selected" : ""}`} onClick={() => setConversationId(conversation.id)}><MessageCircle size={15} /><span><b>{conversation.title}</b><small>Version {conversation.version_number}</small></span></button>)}
          </div>
          {selected && <button className="delete-button" onClick={handleDelete}><Trash2 size={15} /> Delete document</button>}
          <div className="sidebar-footer"><BookOpen size={16} /> Answers stay grounded in your notes.</div>
        </aside>
        <section className="workspace">
          {!selected ? <div className="welcome"><div className="welcome-icon"><Sparkles size={30} /></div><h1>Study smarter with your notes.</h1><p>Upload a PDF to chat with your material, create summaries, and generate flashcards.</p><label className="primary-button"><Upload size={17} /> Upload your first PDF<input type="file" accept="application/pdf" onChange={handleUpload} hidden /></label></div> :
            <><div className="document-header"><div><p className="eyebrow">ACTIVE DOCUMENT</p><h1>{selected.filename}</h1><span>{selected.page_count} pages · {pinnedVersion ? `Version ${pinnedVersion.version_number} pinned` : "current version"}</span></div><div className="document-badge"><span className="status-dot" /> Ready</div></div>
              {(newVersionNotice || pinnedVersionIsOutdated) && <div className="version-notice"><div><strong>New version available: {latestVersion.name || latestVersion.filename} v{latestVersion.version_number}</strong>{pinnedVersionIsOutdated && <p>This conversation is using {selected.name || selected.filename} v{pinnedVersion.version_number}. A newer version is available.</p>}</div><div className="version-actions"><button className="secondary-button" onClick={() => setNewVersionNotice(null)}>Continue with v{pinnedVersion?.version_number || latestVersion.version_number}</button><button className="primary-button" onClick={startNewChatWithLatest} disabled={busy}>Start new chat with v{latestVersion.version_number}</button><button className="secondary-button" disabled title="Version comparison is coming soon">Compare versions</button></div></div>}
              <nav className="tabs"><button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}><MessageCircle size={16} /> Chat</button><button className={pdfOpen ? "active" : ""} onClick={openPdf}><Eye size={16} /> Read PDF</button><button className={tab === "summary" ? "active" : ""} onClick={() => setTab("summary")}><WandSparkles size={16} /> Summary</button><button className={tab === "flashcards" ? "active" : ""} onClick={() => setTab("flashcards")}><Sparkles size={16} /> Flashcards</button><button className={tab === "quiz" ? "active" : ""} onClick={() => setTab("quiz")}><ClipboardCheck size={16} /> Quiz</button></nav>
              {error && <div className="error">{error}</div>}
              {tab === "chat" && <div className={pdfOpen ? "chat-layout split" : "chat-layout"}><div className="chat-panel"><div className="messages">{!messages.length && <div className="chat-empty"><div className="empty-orb"><MessageCircle size={25} /></div><p className="eyebrow">YOUR STUDY COPILOT</p><h2>What would you like to learn?</h2><p>Ask a question, clarify a concept, or use one of the suggestions below.</p><div className="prompt-grid"><button onClick={() => setQuestion("Give me a concise overview of these notes.")}>Give me an overview</button><button onClick={() => setQuestion("What are the most important concepts?")}>Key concepts</button><button onClick={() => setQuestion("Explain the hardest topic simply.")}>Explain simply</button></div></div>}{messages.map((message, index) => <div key={index} className={`message ${message.role}`}><div className="avatar">{message.role === "user" ? "Y" : "N"}</div><div className="message-card"><span className="message-label">{message.role === "user" ? "You" : "NoteBot"}</span><p>{message.content || "Thinking..."}</p>{message.citations?.length > 0 && <details className="sources"><summary>Sources <span>{message.citations.length}</span></summary><div className="source-list">{message.citations.map((citation) => <div className="source-entry" key={`${citation.chunk_id}-${citation.page_number}`}><strong>Source: {citation.document_name}</strong><span>Version: {citation.version_number}</span><span>Page: {citation.page_number}</span></div>)}</div></details>}</div></div>)}</div><form className="composer" onSubmit={handleChat}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your notes..." disabled={busy} /><button disabled={busy || !question.trim()}><Sparkles size={17} /></button></form></div>{pdfOpen && <aside className="pdf-side-panel"><div className="pdf-side-header"><div><strong>Document</strong><span>{selected.filename} · Version {selected.version_number}</span></div><button className="pdf-close" onClick={() => setPdfOpen(false)} aria-label="Close PDF">×</button></div>{pdfError ? <div className="pdf-missing"><Eye size={28} /><h3>PDF preview unavailable</h3><p>{pdfError}</p><label className="primary-button"><Upload size={15} /> Re-upload PDF<input type="file" accept="application/pdf" onChange={handleUpload} hidden /></label></div> : <iframe className="pdf-viewer" title={`Preview of ${selected.filename}`} src={documentFileUrl(selected.id)} />}</aside>}</div>}
              {tab === "summary" && <div className="tool-panel"><div className="tool-title"><WandSparkles size={22} /><div><h2>Document summary</h2><p>Turn your notes into a concise study guide.</p></div></div><button className="primary-button" onClick={runSummary} disabled={busy}>{busy ? "Analyzing..." : "Generate summary"}</button>{assetHistory("summary").length > 0 && <div className="asset-history"><strong>Saved summaries</strong>{assetHistory("summary").map((asset) => <button className={asset.id === savedAssetId ? "asset-item active" : "asset-item"} key={asset.id} onClick={() => openSavedAsset(asset)}>Summary — {asset.document_name} v{asset.version_number}<small>{new Date(asset.created_at).toLocaleString()}</small></button>)}</div>}{summary && <div className="result"><h2>{summary.title}</h2><p>{summary.summary}</p><h3>Key topics</h3><ul>{summary.key_topics.map((item) => <li key={item}>{item}</li>)}</ul><h3>Study tips</h3><ul>{summary.study_tips.map((item) => <li key={item}>{item}</li>)}</ul></div>}</div>}
              {tab === "flashcards" && <div className="tool-panel"><div className="tool-title"><Sparkles size={22} /><div><h2>Flashcards</h2><p>Practice the most important concepts in your notes.</p></div></div><div className="control-row"><label>Cards <select value={cardCount} onChange={(event) => setCardCount(Number(event.target.value))}>{[3, 5, 8, 10, 15].map((count) => <option key={count}>{count}</option>)}</select></label><button className="primary-button" onClick={runFlashcards} disabled={busy}>{busy ? "Creating..." : "Generate cards"}</button></div>{assetHistory("flashcards").length > 0 && <div className="asset-history"><strong>Saved flashcard sets</strong>{assetHistory("flashcards").map((asset) => <button className={asset.id === savedAssetId ? "asset-item active" : "asset-item"} key={asset.id} onClick={() => openSavedAsset(asset)}>Flashcards — {asset.document_name} v{asset.version_number}<small>{new Date(asset.created_at).toLocaleString()}</small></button>)}</div>}<div className="cards">{flashcards.map((card, index) => <article className="card" key={index}><span className={`difficulty ${card.difficulty}`}>{card.difficulty}</span><h3>{card.question}</h3><div className="answer">{card.answer}</div></article>)}</div></div>}
              {tab === "quiz" && <div className="tool-panel"><div className="tool-title"><ClipboardCheck size={22} /><div><h2>Quiz</h2><p>Check your understanding with multiple-choice and short-answer questions.</p></div></div>{!quiz && <button className="primary-button" onClick={runQuiz} disabled={busy}>{busy ? "Creating..." : "Generate quiz"}</button>}{assetHistory("quiz").length > 0 && <div className="asset-history"><strong>Saved quizzes</strong>{assetHistory("quiz").map((label) => <span key={label}>Quiz — {label}</span>)}</div>}{quiz && <form onSubmit={submitQuiz}><div className="quiz-list">{quiz.questions.map((item, index) => <article className="result quiz-question" key={index}><span className={`difficulty ${item.difficulty}`}>{item.difficulty} · {item.question_type.replace("_", " ")}</span><h3>{index + 1}. {item.question}</h3>{item.question_type === "multiple_choice" ? item.options.map((option) => <label className="quiz-option" key={option}><input type="radio" name={`question-${index}`} value={option} checked={quizAnswers[index] === option} onChange={(event) => setQuizAnswers({ ...quizAnswers, [index]: event.target.value })} /> {option}</label>) : <input className="short-answer" value={quizAnswers[index] || ""} onChange={(event) => setQuizAnswers({ ...quizAnswers, [index]: event.target.value })} placeholder="Your answer" />}<small>Source pages: {item.source_pages.join(", ")}</small></article>)}</div><button className="primary-button" type="submit" disabled={busy}>{busy ? "Scoring..." : "Submit quiz"}</button>{quizScore && <div className="result"><h2>Score: {quizScore.score}/{quizScore.total}</h2><p>{quizScore.percentage}% correct</p></div>}</form>}</div>}
            </>}
        </section>
      </main>
    </div>
  );
}

export default App;
