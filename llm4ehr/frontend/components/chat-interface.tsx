"use client"

import { useState, useRef, useEffect } from "react"
import type { ReactNode } from "react"
import { useAppStore, type RetrievalScope, type SourceDocument } from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Send,
  Square,
  User,
  Bot,
  Sparkles,
  FileText,
  Link,
  ChevronDown,
  Search,
} from "lucide-react"
import { cn } from "@/lib/utils"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1"

const RETRIEVAL_SCOPE_OPTIONS: Array<{ value: RetrievalScope; label: string }> = [
  { value: "relevant_papers", label: "Relevant papers" },
  { value: "all_papers", label: "All indexed papers" },
  { value: "specific_paper", label: "Specific paper" },
]

function getRetrievalScopeLabel(scope?: RetrievalScope | null) {
  return RETRIEVAL_SCOPE_OPTIONS.find((option) => option.value === scope)?.label
}

function replaceArticleTitlePlaceholder(content: string, sources?: SourceDocument[]) {
  const articleTitle = sources?.[0]?.title?.trim()
  if (!articleTitle) {
    return content
  }

  return content.replace(/\[Article Title\]/g, articleTitle)
}

function formatSourceScore(score?: number | null) {
  if (typeof score !== "number" || !Number.isFinite(score)) {
    return null
  }

  const boundedScore = Math.min(Math.max(score, 0), 1)
  return `${Math.round(boundedScore * 100)}%`
}

function splitMarkdownTableRow(line: string) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "")
  const cells: string[] = []
  let current = ""
  let escaped = false

  for (const character of trimmed) {
    if (escaped) {
      current += character
      escaped = false
      continue
    }
    if (character === "\\") {
      escaped = true
      continue
    }
    if (character === "|") {
      cells.push(current.trim())
      current = ""
      continue
    }
    current += character
  }
  cells.push(current.trim())

  return cells
}

function isMarkdownTableSeparator(line: string) {
  const cells = splitMarkdownTableRow(line)
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

function hasMarkdownTable(content: string) {
  const lines = content.split("\n")
  return lines.some((line, index) => line.includes("|") && isMarkdownTableSeparator(lines[index + 1] ?? ""))
}

function renderInlineMarkdown(value: string) {
  const segments: ReactNode[] = []
  const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = linkPattern.exec(value)) !== null) {
    if (match.index > lastIndex) {
      segments.push(value.slice(lastIndex, match.index))
    }
    segments.push(
      <a
        key={`${match[2]}-${match.index}`}
        href={match[2]}
        target="_blank"
        rel="noopener noreferrer"
        className="font-medium text-primary underline-offset-2 hover:underline"
      >
        {match[1].replace(/\\([\[\]])/g, "$1")}
      </a>
    )
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < value.length) {
    segments.push(value.slice(lastIndex))
  }

  return segments
}

function renderCellText(value: string) {
  return value.split(/<br\s*\/?>/i).map((part, index, parts) => (
    <span key={`${part}-${index}`}>
      {renderInlineMarkdown(part)}
      {index < parts.length - 1 && <br />}
    </span>
  ))
}

function getTableColumnClass(headers: string[], columnIndex: number) {
  const header = headers[columnIndex]?.trim().toLowerCase()
  if (header === "answer") {
    return "min-w-[28rem] max-w-[44rem] md:w-[55%]"
  }
  if (header === "paper") {
    return "min-w-48 max-w-72"
  }
  return "min-w-36 max-w-80"
}

function MarkdownMessageContent({ content }: { content: string }) {
  const lines = content.split("\n")
  const blocks: Array<
    | { type: "text"; value: string }
    | { type: "table"; headers: string[]; rows: string[][] }
  > = []

  let textBuffer: string[] = []
  let index = 0

  const flushText = () => {
    const value = textBuffer.join("\n").trim()
    if (value) {
      blocks.push({ type: "text", value })
    }
    textBuffer = []
  }

  while (index < lines.length) {
    const current = lines[index]
    const next = lines[index + 1]
    if (
      current?.includes("|") &&
      next &&
      isMarkdownTableSeparator(next)
    ) {
      flushText()
      const headers = splitMarkdownTableRow(current)
      const rows: string[][] = []
      index += 2

      while (index < lines.length && lines[index].includes("|")) {
        const row = splitMarkdownTableRow(lines[index])
        if (row.length === headers.length) {
          rows.push(row)
          index += 1
          continue
        }
        break
      }

      blocks.push({ type: "table", headers, rows })
      continue
    }

    textBuffer.push(current)
    index += 1
  }
  flushText()

  return (
    <div className="space-y-3 text-sm leading-relaxed">
      {blocks.map((block, blockIndex) => {
        if (block.type === "text") {
          return (
            <p key={`text-${blockIndex}`} className="whitespace-pre-wrap">
              {block.value}
            </p>
          )
        }

        return (
          <div
            key={`table-${blockIndex}`}
            className="w-full max-w-full overflow-x-auto rounded-md border border-border/60 bg-background/40"
          >
            <Table className="min-w-[72rem]">
              <TableHeader>
                <TableRow>
                  {block.headers.map((header, headerIndex) => (
                    <TableHead
                      key={`${header}-${headerIndex}`}
                      className={cn(
                        "h-9 bg-muted/70 px-3 text-xs font-semibold whitespace-normal",
                        getTableColumnClass(block.headers, headerIndex),
                      )}
                    >
                      {header}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {block.rows.map((row, rowIndex) => (
                  <TableRow key={`row-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <TableCell
                        key={`cell-${rowIndex}-${cellIndex}`}
                        className={cn(
                          "px-3 py-2 align-top text-xs whitespace-normal",
                          getTableColumnClass(block.headers, cellIndex),
                        )}
                      >
                        {renderCellText(cell)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )
      })}
    </div>
  )
}

function MessageBubble({ message }: { message: { role: 'user' | 'assistant'; content: string; timestamp: Date; sources?: SourceDocument[]; retrievalScope?: RetrievalScope } }) {
  const [sourceExpanded, setSourceExpanded] = useState(false)
  const isUser = message.role === 'user'
  const hasSources = !isUser && message.sources && message.sources.length > 0
  const displayContent = isUser ? message.content : replaceArticleTitlePlaceholder(message.content, message.sources)
  const hasTable = !isUser && hasMarkdownTable(displayContent)
  const shouldScrollSources = (message.sources?.length ?? 0) > 5
  const retrievalScopeLabel = !isUser ? getRetrievalScopeLabel(message.retrievalScope) : null

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
        "flex flex-col gap-1",
        hasTable ? "w-full max-w-[calc(100%-2.75rem)]" : "max-w-[80%]",
        isUser && "items-end"
      )}>
        <div className={cn(
          "rounded-2xl px-4 py-2.5",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : "bg-muted text-foreground rounded-tl-sm"
        )}>
          <MarkdownMessageContent content={displayContent} />
          {retrievalScopeLabel && (
            <Badge variant="outline" className="mt-3 text-xs">
              <Search className="h-3 w-3 mr-1" />
              {retrievalScopeLabel}
            </Badge>
          )}
          {hasSources && (
            <div className="mt-3 border-t border-border/50 pt-3">
              <button
                onClick={() => setSourceExpanded(!sourceExpanded)}
                className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
              >
                <span className="font-semibold">Sources ({message.sources!.length})</span>
                <ChevronDown className={cn(
                  "h-3.5 w-3.5 transition-transform",
                  sourceExpanded && "rotate-180"
                )} />
              </button>
              {sourceExpanded && (
                <div className={cn("mt-2 space-y-2", shouldScrollSources && "max-h-64 overflow-y-auto pr-1")}>
                  {message.sources!.map((source) => {
                    const sourceScore = formatSourceScore(source.score)
                    return (
                      <div
                        key={`${source.article_id}-${source.title}`}
                        className="text-xs bg-background/50 rounded-lg p-2.5 border border-border/30 hover:border-border/60 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-foreground line-clamp-2 text-xs">{source.title}</p>
                            <div className="flex items-center gap-2 mt-1 flex-wrap">
                              <span className="font-mono text-xs text-muted-foreground">{source.article_id}</span>
                              {sourceScore && (
                                <Badge variant="outline" className="text-xs py-0 px-1.5">
                                  {sourceScore}
                                </Badge>
                              )}
                            </div>
                          </div>
                          {source.url && (
                            <a
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-primary hover:underline shrink-0 whitespace-nowrap ml-2"
                            >
                              <Link className="h-3.5 w-3.5" />
                            </a>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
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
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const {
    messages,
    isGenerating,
    addMessage,
    updateMessage,
    setIsGenerating,
    selectedChatModel,
    useHostedModels,
    hostedProvider,
    hostedModelName,
    hostedApiKey,
    temperature,
    maxTokens,
    topP,
    retrievalScope,
    setRetrievalScope,
  } = useAppStore()

  const hasModel = useHostedModels
    ? Boolean(hostedModelName.trim() && hostedApiKey.trim())
    : Boolean(selectedChatModel)

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

  const streamAssistantMessage = (
    messageId: string,
    fullText: string,
    sources?: SourceDocument[],
    responseRetrievalScope?: RetrievalScope,
  ) => {
    if (!fullText) {
      updateMessage(messageId, {
        content: "No response received.",
        sources,
        retrievalScope: responseRetrievalScope,
      })
      return
    }

    let index = 0
    const chunkSize = 6
    const timer = setInterval(() => {
      index = Math.min(fullText.length, index + chunkSize)
      updateMessage(messageId, {
        content: fullText.slice(0, index),
        sources,
        retrievalScope: responseRetrievalScope,
      })
      if (index >= fullText.length) {
        clearInterval(timer)
        setIsGenerating(false)
      }
    }, 20)
  }

  const handleSubmit = async () => {
    if (!input.trim() || isGenerating || !hasModel) return
    setErrorMessage(null)

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

    const requestMessages = [...messages, userMessage].map((message) => ({
      role: message.role,
      content: message.content,
    }))

    const controller = new AbortController()
    abortRef.current = controller

    const assistantMessageId = crypto.randomUUID()
    addMessage({
      id: assistantMessageId,
      role: 'assistant',
      content: "",
      timestamp: new Date(),
    })

    let didStream = false

    try {
      const endpoint = useHostedModels ? `${API_BASE}/rag/remote` : `${API_BASE}/rag/local`
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: requestMessages,
          retrieval_scope: retrievalScope,
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          ...(useHostedModels
            ? {
              hosted_provider: hostedProvider,
              hosted_model_name: hostedModelName.trim(),
              hosted_api_key: hostedApiKey.trim(),
            }
            : {}),
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Chat request failed")
      }

      const data = await response.json()
      didStream = true
      streamAssistantMessage(
        assistantMessageId,
        data.response ?? "",
        data.source_documents ?? [],
        data.retrieval_scope ?? retrievalScope,
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return
      }
      const message = error instanceof Error ? error.message : "Chat request failed"
      setErrorMessage(message)
      updateMessage(assistantMessageId, {
        content: `Error: ${message}`,
      })
      setIsGenerating(false)
    } finally {
      if (!didStream) {
        setIsGenerating(false)
      }
      abortRef.current = null
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setIsGenerating(false)
  }

  const handleRetrievalScopeChange = (value: string) => {
    setRetrievalScope(value as RetrievalScope)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
    })
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
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            Search scope
          </span>
          <Select
            value={retrievalScope}
            onValueChange={handleRetrievalScopeChange}
            disabled={isGenerating}
          >
            <SelectTrigger size="sm" className="w-[190px]">
              <Search className="h-3.5 w-3.5" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RETRIEVAL_SCOPE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
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
              disabled={!input.trim() || !hasModel}
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          )}
        </div>
        {!hasModel && (
          <p className="text-xs text-amber-600 dark:text-amber-400 text-center mt-2">
            {useHostedModels
              ? "Choose a hosted chat model and API key in Settings → Models before sending messages."
              : "Select a local chat model in Settings → Models before sending messages."}
          </p>
        )}
        {errorMessage && (
          <p className="text-xs text-destructive text-center mt-2">
            {errorMessage}
          </p>
        )}
        <p className="text-xs text-muted-foreground text-center mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
