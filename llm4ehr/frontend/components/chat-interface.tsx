"use client"

import { useState, useRef, useEffect } from "react"
import { useAppStore } from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  Send,
  Square,
  User,
  Bot,
  Sparkles,
  AlertCircle,
  FileText,
  Link
} from "lucide-react"
import { cn } from "@/lib/utils"

function MessageBubble({ message }: { message: { role: 'user' | 'assistant'; content: string; timestamp: Date } }) {
  const isUser = message.role === 'user'

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
        isUser ? "bg-primary" : "bg-muted"
      )}>
        {isUser ? (
          <User className="h-4 w-4 text-primary-foreground" />
        ) : (
          <Bot className="h-4 w-4 text-muted-foreground" />
        )}
      </div>
      <div className={cn(
        "flex flex-col gap-1 max-w-[80%]",
        isUser && "items-end"
      )}>
        <div className={cn(
          "rounded-2xl px-4 py-2.5",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : "bg-muted text-foreground rounded-tl-sm"
        )}>
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
        <span className="text-xs text-muted-foreground px-1">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 mb-4">
        <Sparkles className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-xl font-semibold mb-2">Welcome to LLM4EHR</h2>
      <p className="text-muted-foreground max-w-md mb-6">
        Your AI-powered assistant for healthcare research. Configure your models and context sources in settings, then start chatting.
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        <Badge variant="secondary" className="text-xs">
          <FileText className="h-3 w-3 mr-1" />
          Upload PDFs
        </Badge>
        <Badge variant="secondary" className="text-xs">
          <Link className="h-3 w-3 mr-1" />
          Nature Papers
        </Badge>
        <Badge variant="secondary" className="text-xs">
          <Bot className="h-3 w-3 mr-1" />
          Local Models
        </Badge>
      </div>
    </div>
  )
}

export function ChatInterface() {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  const {
    messages,
    isGenerating,
    addMessage,
    setIsGenerating,
    selectedChatModel,
    useHostedModels,
    hostedModelName,
    contextSources,
  } = useAppStore()

  const hasModel = selectedChatModel || (useHostedModels && hostedModelName)
  const activeModelName = useHostedModels ? hostedModelName : selectedChatModel
  const hasContext = contextSources.length > 0

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesContainerRef.current) {
      // Use setTimeout to ensure the DOM has updated
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
        }
      }, 0)
    }
  }, [messages, isGenerating])  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleSubmit = async () => {
    if (!input.trim() || isGenerating) return

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user' as const,
      content: input.trim(),
      timestamp: new Date(),
    }

    addMessage(userMessage)
    setInput('')
    setIsGenerating(true)

    // Keep focus on textarea while generating
    if (textareaRef.current) {
      textareaRef.current.focus()
    }

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: 'assistant' as const,
        content: `Thank you for your question about "${input.trim().slice(0, 50)}..."\n\nThis is a simulated response from the LLM4EHR assistant. In a production environment, this would connect to your selected model (${activeModelName || 'No model selected'}) and process your query against the loaded context sources.\n\nYou have ${contextSources.length} context source${contextSources.length !== 1 ? 's' : ''} available for reference.`,
        timestamp: new Date(),
      }
      addMessage(assistantMessage)
      setIsGenerating(false)
    }, 1500)
  }

  const handleStop = () => {
    setIsGenerating(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      {/* Messages Area */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <div
          ref={messagesContainerRef}
          className="h-full overflow-y-auto"
        >
          <div className="p-4">
            {messages.length === 0 ? (
              <EmptyState />
            ) : (
              <div className="space-y-6">
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {isGenerating && (
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                      <Bot className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl rounded-tl-sm bg-muted">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border p-4 bg-background shrink-0">
        <div className="flex gap-2 items-end w-full">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about research papers, EHR data, or medical literature..."
              className="min-h-[44px] max-h-[200px] resize-none pr-12 py-3"
              rows={1}
            />
          </div>
          {isGenerating ? (
            <Button
              variant="destructive"
              size="icon"
              className="h-11 w-11 shrink-0"
              onClick={handleStop}
            >
              <Square className="h-4 w-4" />
              <span className="sr-only">Stop generating</span>
            </Button>
          ) : (
            <Button
              size="icon"
              className="h-11 w-11 shrink-0"
              onClick={handleSubmit}
              disabled={!input.trim()}
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          )}
        </div>
        <p className="text-xs text-muted-foreground text-center mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
