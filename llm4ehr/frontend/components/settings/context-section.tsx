"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { useAppStore, type ContextSource } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  FileText,
  Trash2,
  Plus,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Info,
  BookOpen,
  Files,
  X,
  Database,
  RefreshCw
} from "lucide-react"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1"

type FetchedArticle = {
  article_id: string
  title?: string | null
  url: string
}

export function ContextSection() {
  const {
    contextSources,
    indexingState,
    selectedEmbeddingModel,
    addContextSource,
    removeContextSource,
    updateContextSource,
    startIndexing,
    completeIndexing,
    stopIndexing,
    setNeedsReindex,
  } = useAppStore()

  const [isDragging, setIsDragging] = useState(false)
  const [indexingProgress, setIndexingProgress] = useState(0)
  const [indexingMessage, setIndexingMessage] = useState<string | null>(null)
  const [indexingError, setIndexingError] = useState<string | null>(null)
  const [fetchQuery, setFetchQuery] = useState("")
  const [fetchMaxResults, setFetchMaxResults] = useState("20")
  const [resetCursorQuery, setResetCursorQuery] = useState("")
  const [batchUrls, setBatchUrls] = useState("")
  const [fetchStatus, setFetchStatus] = useState<string | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [isFetching, setIsFetching] = useState(false)
  const [isListingFetched, setIsListingFetched] = useState(false)
  const [fetchedArticles, setFetchedArticles] = useState<FetchedArticle[]>([])
  const [selectedArticleIds, setSelectedArticleIds] = useState<string[]>([])
  const [indexingTarget, setIndexingTarget] = useState<"context" | "fetched">("context")
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Track previous source count to detect new additions
  const prevSourceCountRef = useRef(contextSources.length)

  useEffect(() => {
    // If new sources were added, mark as needing reindex
    if (contextSources.length > prevSourceCountRef.current && indexingState.lastIndexedAt) {
      setNeedsReindex(true)
    }
    prevSourceCountRef.current = contextSources.length
  }, [contextSources.length, indexingState.lastIndexedAt, setNeedsReindex])

  const handleStartIndexing = async () => {
    setIndexingError(null)
    setIndexingMessage(null)
    setIndexingProgress(0)
    setIndexingTarget("context")
    startIndexing()
    try {
      const response = await fetch(`${API_BASE}/index/all`, { method: "POST" })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Indexing failed")
      }
      const data = await response.json()
      setIndexingMessage(`Indexed ${data.indexed_count ?? 0} documents`)
      setIndexingProgress(100)
      completeIndexing()
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : "Indexing failed")
      stopIndexing()
    }
  }

  const handleStopIndexing = async () => {
    setIndexingError(null)
    setIndexingMessage(null)
    try {
      const response = await fetch(`${API_BASE}/index/stop`, { method: "POST" })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Stop indexing failed")
      }
      const data = await response.json()
      setIndexingMessage(data.message ?? "Indexing stop requested")
      stopIndexing()
      setIndexingProgress(0)
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : "Stop indexing failed")
      stopIndexing()
    }
  }

  const handleDeleteIndex = async (deleteAll: boolean) => {
    setIndexingError(null)
    setIndexingMessage(null)
    try {
      const body = deleteAll
        ? { delete_all: true, article_ids: [] }
        : { delete_all: false, article_ids: selectedArticleIds }
      const response = await fetch(`${API_BASE}/index`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Delete failed")
      }
      const data = await response.json()
      setIndexingMessage(`Deleted ${data.deleted_count ?? 0} indexed documents`)
      stopIndexing()
      setIndexingProgress(0)
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : "Delete failed")
      stopIndexing()
    }
  }

  const handleIndexSelected = async () => {
    if (selectedArticleIds.length === 0) return
    setIndexingError(null)
    setIndexingMessage(null)
    setIndexingProgress(0)
    startIndexing()
    setIndexingTarget("fetched")
    try {
      const response = await fetch(`${API_BASE}/index/articles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article_ids: selectedArticleIds }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Indexing failed")
      }
      const data = await response.json()
      setIndexingMessage(`Indexed ${data.indexed_count ?? 0} documents`)
      setIndexingProgress(100)
      completeIndexing()
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : "Indexing failed")
      stopIndexing()
    }
  }

  const handleIndexAllFetched = async () => {
    if (fetchedArticles.length === 0) return
    const articleIds = fetchedArticles.map((article) => article.article_id)
    setIndexingError(null)
    setIndexingMessage(null)
    setIndexingProgress(0)
    startIndexing()
    setIndexingTarget("fetched")
    try {
      const response = await fetch(`${API_BASE}/index/articles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article_ids: articleIds }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Indexing failed")
      }
      const data = await response.json()
      setIndexingMessage(`Indexed ${data.indexed_count ?? 0} documents`)
      setIndexingProgress(100)
      completeIndexing()
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : "Indexing failed")
      stopIndexing()
    }
  }

  const loadFetchedArticles = useCallback(async () => {
    setIsListingFetched(true)
    setFetchError(null)
    try {
      const response = await fetch(`${API_BASE}/fetch/list`)
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Failed to list fetched articles")
      }
      const data = await response.json()
      setFetchedArticles(data.articles ?? [])
    } catch (error) {
      setFetchError(error instanceof Error ? error.message : "Failed to list fetched articles")
    } finally {
      setIsListingFetched(false)
    }
  }, [])

  useEffect(() => {
    loadFetchedArticles()
  }, [loadFetchedArticles])

  const handleFetchFromOpenAlex = async () => {
    if (!fetchQuery.trim()) return
    setIsFetching(true)
    setFetchError(null)
    setFetchStatus(null)
    try {
      const response = await fetch(`${API_BASE}/fetch/from-openalex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: fetchQuery.trim(),
          max_results: Number(fetchMaxResults) || 20,
        }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Fetch failed")
      }
      setFetchStatus("Fetch completed. Refresh the list to view articles.")
      await loadFetchedArticles()
    } catch (error) {
      setFetchError(error instanceof Error ? error.message : "Fetch failed")
    } finally {
      setIsFetching(false)
    }
  }

  const handleResetCursor = async () => {
    if (!resetCursorQuery.trim()) return
    setIsFetching(true)
    setFetchError(null)
    setFetchStatus(null)
    const queryTerms = resetCursorQuery
      .split(",")
      .map((term) => term.trim())
      .filter(Boolean)
    try {
      const response = await fetch(`${API_BASE}/fetch/reset-cursor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(queryTerms),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Reset cursor failed")
      }
      const data = await response.json()
      setFetchStatus(data.message ?? "Cursor reset")
    } catch (error) {
      setFetchError(error instanceof Error ? error.message : "Reset cursor failed")
    } finally {
      setIsFetching(false)
    }
  }

  const handleBatchFetch = async () => {
    const urls = batchUrls
      .split(/[\n,\s]+/)
      .map((url) => url.trim())
      .filter(Boolean)
    if (urls.length === 0) return
    setIsFetching(true)
    setFetchError(null)
    setFetchStatus(null)
    try {
      const response = await fetch(`${API_BASE}/fetch/batch?save=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Batch fetch failed")
      }
      setFetchStatus(`Fetched ${urls.length} article(s).`)
      setBatchUrls("")
      await loadFetchedArticles()
    } catch (error) {
      setFetchError(error instanceof Error ? error.message : "Batch fetch failed")
    } finally {
      setIsFetching(false)
    }
  }

  const toggleArticleSelection = (articleId: string) => {
    setSelectedArticleIds((prev) =>
      prev.includes(articleId)
        ? prev.filter((id) => id !== articleId)
        : [...prev, articleId]
    )
  }

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return

    const pdfFiles = Array.from(files).filter(
      file => file.type === "application/pdf" || file.name.toLowerCase().endsWith('.pdf')
    )

    pdfFiles.forEach((file) => {
      const newSource: ContextSource = {
        id: crypto.randomUUID(),
        type: "pdf",
        name: file.name,
        status: "processing",
      }

      addContextSource(newSource)

      // Simulate processing with varied times based on file size
      const processingTime = Math.min(2000 + (file.size / 100000) * 500, 8000)
      setTimeout(() => {
        updateContextSource(newSource.id, { status: "ready" })
      }, processingTime)
    })
  }, [addContextSource, updateContextSource])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    handleFileUpload(files)
    // Reset input
    e.target.value = ""
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    handleFileUpload(files)
  }, [handleFileUpload])

  const getStatusBadge = (status: ContextSource["status"]) => {
    switch (status) {
      case "pending":
        return <Badge variant="secondary">Pending</Badge>
      case "processing":
        return (
          <Badge variant="secondary" className="gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            Processing
          </Badge>
        )
      case "ready":
        return (
          <Badge variant="default" className="gap-1 bg-accent text-accent-foreground">
            <CheckCircle2 className="h-3 w-3" />
            Ready
          </Badge>
        )
      case "error":
        return (
          <Badge variant="destructive" className="gap-1">
            <AlertCircle className="h-3 w-3" />
            Error
          </Badge>
        )
    }
  }

  const urlSources = contextSources.filter(s => s.type === "url")
  const pdfSources = contextSources.filter(s => s.type === "pdf")
  const indexingSourceCount = indexingTarget === "fetched"
    ? (selectedArticleIds.length || fetchedArticles.length)
    : contextSources.length

  const clearAllPdfs = () => {
    pdfSources.forEach(s => removeContextSource(s.id))
  }

  return (
    <div className="space-y-6">
      {/* Indexing Card - Moved to top for visibility */}
      <Card className={`${indexingState.needsReindex ? "border-amber-500/50 ring-2 ring-amber-500/20" : contextSources.length > 0 && !indexingState.lastIndexedAt ? "border-primary/50 ring-2 ring-primary/20" : ""}`}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Document Indexing
            {indexingState.needsReindex && (
              <Badge variant="secondary" className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30">
                Action Required
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            Index your documents to enable context-aware conversations. Re-indexing is required when you change the embedding model or add new sources.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className={`rounded-lg border p-4 ${indexingState.needsReindex
            ? "border-amber-500/50 bg-amber-500/5"
            : indexingState.lastIndexedAt
              ? "border-accent/50 bg-accent/5"
              : "border-border"
            }`}>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  {indexingState.isIndexing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span className="font-medium">Indexing in progress...</span>
                    </>
                  ) : indexingState.needsReindex ? (
                    <>
                      <RefreshCw className="h-4 w-4 text-amber-500" />
                      <span className="font-medium text-amber-600 dark:text-amber-400">Re-indexing required</span>
                    </>
                  ) : indexingState.lastIndexedAt ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-accent" />
                      <span className="font-medium">Index up to date</span>
                    </>
                  ) : contextSources.length > 0 ? (
                    <>
                      <AlertCircle className="h-4 w-4 text-primary" />
                      <span className="font-medium text-primary">Ready to index</span>
                    </>
                  ) : (
                    <>
                      <Info className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium text-muted-foreground">No documents added yet</span>
                    </>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">
                  {indexingState.isIndexing ? (
                    <span>
                      Processing {indexingSourceCount} {indexingTarget === "fetched" ? "fetched article" : "source"}
                      {indexingSourceCount !== 1 ? 's' : ''}...
                    </span>
                  ) : indexingState.lastIndexedAt ? (
                    <span>
                      Last indexed: {new Date(indexingState.lastIndexedAt).toLocaleString()}
                      {indexingState.lastEmbeddingModel && (
                        <span className="ml-2 font-mono text-xs">({indexingState.lastEmbeddingModel})</span>
                      )}
                    </span>
                  ) : contextSources.length > 0 ? (
                    <span>{contextSources.length} source{contextSources.length !== 1 ? 's' : ''} ready for indexing</span>
                  ) : (
                    <span>Add Nature URLs or upload PDFs below to get started</span>
                  )}
                </div>
              </div>

              <Button
                onClick={handleStartIndexing}
                disabled={indexingState.isIndexing}
                size="lg"
                className={`min-w-[160px] ${indexingState.needsReindex ? "bg-amber-600 hover:bg-amber-700" : ""}`}
              >
                {indexingState.isIndexing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Indexing...
                  </>
                ) : indexingState.needsReindex ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Re-Index Now
                  </>
                ) : (
                  <>
                    <Database className="h-4 w-4 mr-2" />
                    Start Indexing
                  </>
                )}
              </Button>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Button variant="outline" onClick={handleIndexAllFetched} disabled={fetchedArticles.length === 0 || indexingState.isIndexing}>
                Index All Fetched
              </Button>
              <Button variant="outline" onClick={handleStopIndexing} disabled={!indexingState.isIndexing}>
                Stop Indexing
              </Button>
              <Button variant="outline" onClick={() => handleDeleteIndex(false)} disabled={selectedArticleIds.length === 0 || indexingState.isIndexing}>
                Delete Selected Index
              </Button>
              <Button variant="destructive" onClick={() => handleDeleteIndex(true)} disabled={indexingState.isIndexing}>
                Delete All Index
              </Button>
            </div>

            {indexingState.isIndexing && (
              <div className="mt-4 space-y-2">
                <Progress value={indexingProgress} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {Math.round(indexingProgress)}% complete
                </p>
              </div>
            )}

            {indexingMessage && (
              <div className="mt-3 text-sm text-accent-foreground bg-accent/10 border border-accent/30 rounded-lg p-2">
                {indexingMessage}
              </div>
            )}

            {indexingError && (
              <div className="mt-3 text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-lg p-2">
                {indexingError}
              </div>
            )}
          </div>

          {!selectedEmbeddingModel && contextSources.length > 0 && (
            <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 p-3 text-sm">
              <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
              <p className="text-amber-700 dark:text-amber-300">
                Please select an embedding model in the <span className="font-medium">Models</span> tab before indexing your documents.
              </p>
            </div>
          )}

          {/* Quick Stats */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border bg-muted/30 p-3 text-center">
              <div className="text-2xl font-bold text-primary">{urlSources.length}</div>
              <p className="text-xs text-muted-foreground">URLs</p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3 text-center">
              <div className="text-2xl font-bold text-primary">{pdfSources.length}</div>
              <p className="text-xs text-muted-foreground">PDFs</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Fetching Papers
          </CardTitle>
          <CardDescription>
            Fetch articles via OpenAlex or direct URLs, then index the fetched content.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="fetch-query">OpenAlex Query</Label>
            <div className="flex flex-wrap gap-2">
              <Input
                id="fetch-query"
                placeholder="e.g., prostate cancer radiomics"
                value={fetchQuery}
                onChange={(event) => setFetchQuery(event.target.value)}
                className="flex-1 min-w-[220px]"
              />
              <Input
                type="number"
                min="1"
                max="200"
                value={fetchMaxResults}
                onChange={(event) => setFetchMaxResults(event.target.value)}
                className="w-[120px]"
                placeholder="Max"
              />
              <Button onClick={handleFetchFromOpenAlex} disabled={!fetchQuery.trim() || isFetching}>
                {isFetching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                <span className="ml-2">Run Fetcher</span>
              </Button>
            </div>
          </div>

          <div className="space-y-3">
            <Label htmlFor="reset-cursor">Reset Cursor (comma-separated terms)</Label>
            <div className="flex flex-wrap gap-2">
              <Input
                id="reset-cursor"
                placeholder="e.g., prostate, radiomics"
                value={resetCursorQuery}
                onChange={(event) => setResetCursorQuery(event.target.value)}
                className="flex-1 min-w-[220px]"
              />
              <Button variant="outline" onClick={handleResetCursor} disabled={!resetCursorQuery.trim() || isFetching}>
                Reset Cursor
              </Button>
            </div>
          </div>

          <div className="space-y-3">
            <Label htmlFor="batch-urls">Batch Fetch URLs</Label>
            <Textarea
              id="batch-urls"
              placeholder="Paste one URL per line"
              value={batchUrls}
              onChange={(event) => setBatchUrls(event.target.value)}
              className="min-h-[120px] font-mono text-sm"
            />
            <Button onClick={handleBatchFetch} disabled={!batchUrls.trim() || isFetching}>
              Fetch URLs
            </Button>
          </div>

          {fetchStatus && (
            <div className="text-sm text-accent-foreground bg-accent/10 border border-accent/30 rounded-lg p-2">
              {fetchStatus}
            </div>
          )}

          {fetchError && (
            <div className="text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-lg p-2">
              {fetchError}
            </div>
          )}

          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label>Fetched Articles</Label>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={loadFetchedArticles} disabled={isListingFetched}>
                  {isListingFetched ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                  <span className="ml-2">Refresh</span>
                </Button>
                <Button variant="outline" size="sm" onClick={handleIndexSelected} disabled={selectedArticleIds.length === 0}>
                  Index Selected
                </Button>
              </div>
            </div>
            <div className="max-h-[240px] overflow-y-auto rounded-lg border">
              {fetchedArticles.length === 0 ? (
                <div className="p-4 text-sm text-muted-foreground">No fetched articles yet.</div>
              ) : (
                fetchedArticles.map((article) => (
                  <label
                    key={article.article_id}
                    className="flex items-center gap-3 px-3 py-2 border-b last:border-b-0 text-sm"
                  >
                    <input
                      type="checkbox"
                      checked={selectedArticleIds.includes(article.article_id)}
                      onChange={() => toggleArticleSelection(article.article_id)}
                    />
                    <div className="min-w-0">
                      <p className="font-medium truncate">{article.title ?? article.article_id}</p>
                      <p className="text-xs text-muted-foreground truncate">{article.url}</p>
                    </div>
                  </label>
                ))
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* PDF Upload */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Files className="h-5 w-5" />
            PDF Documents
          </CardTitle>
          <CardDescription>
            Upload multiple PDF documents at once to use as context for your conversations.
            Documents are processed locally and their content is indexed for retrieval.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${isDragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50"
              }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={handleInputChange}
              className="hidden"
              id="pdf-upload"
            />
            <div className="flex flex-col items-center gap-3">
              <div className={`flex h-14 w-14 items-center justify-center rounded-full transition-colors ${isDragging ? "bg-primary/10" : "bg-muted"
                }`}>
                <Upload className={`h-7 w-7 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <div>
                <p className="font-medium">
                  {isDragging ? "Drop PDFs here" : "Click to upload or drag and drop"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Select multiple PDF files at once
                </p>
              </div>
              <p className="text-xs text-muted-foreground">
                Supports bulk upload. Maximum 50MB per file.
              </p>
            </div>
          </div>

          {/* PDF list */}
          {pdfSources.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Uploaded PDFs ({pdfSources.length})</Label>
                <Button variant="ghost" size="sm" onClick={clearAllPdfs} className="h-7 text-xs">
                  <X className="h-3 w-3 mr-1" />
                  Clear All
                </Button>
              </div>
              <div className="max-h-[250px] overflow-y-auto space-y-2 rounded-lg border p-2">
                {pdfSources.map((source) => (
                  <div
                    key={source.id}
                    className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <FileText className="h-4 w-4 shrink-0 text-destructive" />
                      <p className="text-sm font-medium truncate">{source.name}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {getStatusBadge(source.status)}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => removeContextSource(source.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        <span className="sr-only">Remove PDF</span>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
            <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-muted-foreground">
              PDFs are processed using OCR and text extraction. The content is indexed locally
              and used to provide relevant context when you ask questions. Select multiple files
              from your file browser or drag and drop a batch of PDFs.
            </p>
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
