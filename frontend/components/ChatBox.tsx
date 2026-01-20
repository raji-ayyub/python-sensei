"use client";

import { useState } from "react";
import { askQuestion } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatBox() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!input.trim()) return;

    const question = input;
    setInput("");
    setLoading(true);

    setMessages((m) => [...m, { role: "user", content: question }]);

    try {
      const res = await askQuestion(question);
      setMessages((m) => [...m, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: "Something went wrong." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="space-y-4 mb-4 max-h-[60vh] overflow-y-auto">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-lg px-4 py-2 text-sm ${
              m.role === "user"
                ? "bg-blue-600/20 text-blue-200"
                : "bg-neutral-800 text-neutral-200"
            }`}
          >
            {m.content}
          </div>
        ))}

        {loading && (
          <div className="text-neutral-400 text-sm animate-pulse">
            Thinking...
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask a Python question..."
          className="flex-1 rounded-lg bg-neutral-950 border border-neutral-800 px-3 py-2 text-sm focus:outline-none focus:ring focus:ring-blue-500"
        />
        <button
          onClick={send}
          disabled={loading}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
        >
          Ask
        </button>
      </div>
    </div>
  );
}