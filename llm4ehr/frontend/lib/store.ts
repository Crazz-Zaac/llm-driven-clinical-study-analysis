"use client"

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface LocalModel {
  id: string
  name: string
  size: string
  type: 'chat' | 'embedding' | 'vision'
  status: 'available' | 'downloading' | 'downloaded'
  downloadProgress?: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: SourceDocument[]
  retrievalScope?: RetrievalScope
}

export type RetrievalScope = 'relevant_papers' | 'all_papers' | 'specific_paper'

export interface SourceDocument {
  article_id: string
  title: string
  url?: string | null
  score?: number | null
}

export interface ContextSource {
  id: string
  type: 'url' | 'pdf'
  name: string
  url?: string
  status: 'pending' | 'processing' | 'ready' | 'error'
  indexed?: boolean
}

export interface IndexingState {
  isIndexing: boolean
  progress: number
  lastIndexedAt: string | null
  lastEmbeddingModel: string | null
  needsReindex: boolean
}

export type HostedProvider = 'huggingface' | 'openai' | 'mistral'

interface AppState {
  // Model Source
  useHostedModels: boolean

  // Local Model Settings
  selectedChatModel: string | null
  selectedEmbeddingModel: string | null
  selectedVisionModel: string | null
  localModels: LocalModel[]

  // Hosted Model Settings
  hostedModelName: string
  hostedApiKey: string
  hostedProvider: HostedProvider

  // General Model Parameters
  temperature: number
  maxTokens: number
  topP: number

  // Context Sources
  contextSources: ContextSource[]
  natureUrl: string

  // Indexing
  indexingState: IndexingState

  // Chat
  messages: Message[]
  isGenerating: boolean
  retrievalScope: RetrievalScope

  // Actions
  setUseHostedModels: (use: boolean) => void
  setSelectedChatModel: (model: string | null) => void
  setSelectedEmbeddingModel: (model: string | null) => void
  setSelectedVisionModel: (model: string | null) => void
  addLocalModel: (model: LocalModel) => void
  updateLocalModel: (id: string, updates: Partial<LocalModel>) => void
  removeLocalModel: (id: string) => void
  setHostedModelName: (name: string) => void
  setHostedApiKey: (key: string) => void
  setHostedProvider: (provider: HostedProvider) => void
  setTemperature: (temp: number) => void
  setMaxTokens: (tokens: number) => void
  setTopP: (topP: number) => void
  addContextSource: (source: ContextSource) => void
  removeContextSource: (id: string) => void
  updateContextSource: (id: string, updates: Partial<ContextSource>) => void
  setNatureUrl: (url: string) => void
  startIndexing: () => void
  completeIndexing: () => void
  stopIndexing: () => void
  setNeedsReindex: (needs: boolean) => void
  addMessage: (message: Message) => void
  updateMessage: (id: string, updates: Partial<Message>) => void
  clearMessages: () => void
  setIsGenerating: (generating: boolean) => void
  setRetrievalScope: (scope: RetrievalScope) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Model Source
      useHostedModels: false,

      // Local Model Settings
      selectedChatModel: null,
      selectedEmbeddingModel: null,
      selectedVisionModel: null,
      localModels: [],

      // Hosted Model Settings
      hostedModelName: '',
      hostedApiKey: '',
      hostedProvider: 'huggingface',

      // General Model Parameters
      temperature: 0.7,
      maxTokens: 2048,
      topP: 0.9,

      // Context Sources
      contextSources: [],
      natureUrl: '',

      // Indexing
      indexingState: {
        isIndexing: false,
        progress: 0,
        lastIndexedAt: null,
        lastEmbeddingModel: null,
        needsReindex: false,
      },

      // Chat
      messages: [],
      isGenerating: false,
      retrievalScope: 'relevant_papers',

      // Actions
      setUseHostedModels: (use) => set({ useHostedModels: use }),
      setSelectedChatModel: (model) => set({ selectedChatModel: model }),
      setSelectedEmbeddingModel: (model) => set({ selectedEmbeddingModel: model }),
      setSelectedVisionModel: (model) => set({ selectedVisionModel: model }),
      addLocalModel: (model) => set((state) => ({ localModels: [...state.localModels, model] })),
      updateLocalModel: (id, updates) => set((state) => ({
        localModels: state.localModels.map(m => m.id === id ? { ...m, ...updates } : m)
      })),
      removeLocalModel: (id) => set((state) => ({
        localModels: state.localModels.filter(m => m.id !== id),
        selectedChatModel: state.selectedChatModel === id ? null : state.selectedChatModel,
        selectedEmbeddingModel: state.selectedEmbeddingModel === id ? null : state.selectedEmbeddingModel,
        selectedVisionModel: state.selectedVisionModel === id ? null : state.selectedVisionModel,
      })),
      setHostedModelName: (name) => set({ hostedModelName: name }),
      setHostedApiKey: (key) => set({ hostedApiKey: key }),
      setHostedProvider: (provider) => set({ hostedProvider: provider }),
      setTemperature: (temp) => set({ temperature: temp }),
      setMaxTokens: (tokens) => set({ maxTokens: tokens }),
      setTopP: (topP) => set({ topP }),
      addContextSource: (source) => set((state) => ({ contextSources: [...state.contextSources, source] })),
      removeContextSource: (id) => set((state) => ({
        contextSources: state.contextSources.filter(s => s.id !== id)
      })),
      updateContextSource: (id, updates) => set((state) => ({
        contextSources: state.contextSources.map(s => s.id === id ? { ...s, ...updates } : s)
      })),
      setNatureUrl: (url) => set({ natureUrl: url }),
      startIndexing: () => set((state) => ({
        indexingState: { ...state.indexingState, isIndexing: true, progress: 0 }
      })),
      completeIndexing: () => set((state) => ({
        indexingState: {
          ...state.indexingState,
          isIndexing: false,
          progress: 100,
          lastIndexedAt: new Date().toISOString(),
          lastEmbeddingModel: state.selectedEmbeddingModel,
          needsReindex: false,
        },
        contextSources: state.contextSources.map(s => ({ ...s, indexed: true }))
      })),
      stopIndexing: () => set((state) => ({
        indexingState: { ...state.indexingState, isIndexing: false, progress: 0 }
      })),
      setNeedsReindex: (needs) => set((state) => ({
        indexingState: { ...state.indexingState, needsReindex: needs }
      })),
      addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
      updateMessage: (id, updates) => set((state) => ({
        messages: state.messages.map((message) => message.id === id ? { ...message, ...updates } : message)
      })),
      clearMessages: () => set({ messages: [] }),
      setIsGenerating: (generating) => set({ isGenerating: generating }),
      setRetrievalScope: (scope) => set({ retrievalScope: scope }),
    }),
    {
      name: 'llm4ehr-storage',
      partialize: (state) => ({
        useHostedModels: state.useHostedModels,
        selectedChatModel: state.selectedChatModel,
        selectedEmbeddingModel: state.selectedEmbeddingModel,
        selectedVisionModel: state.selectedVisionModel,
        localModels: state.localModels,
        hostedModelName: state.hostedModelName,
        hostedApiKey: state.hostedApiKey,
        hostedProvider: state.hostedProvider,
        temperature: state.temperature,
        maxTokens: state.maxTokens,
        topP: state.topP,
        retrievalScope: state.retrievalScope,
        contextSources: state.contextSources,
        natureUrl: state.natureUrl,
        indexingState: {
          lastIndexedAt: state.indexingState.lastIndexedAt,
          lastEmbeddingModel: state.indexingState.lastEmbeddingModel,
          needsReindex: state.indexingState.needsReindex,
          isIndexing: false,
          progress: 0,
        },
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return
        state.stopIndexing()
      },
    }
  )
)
