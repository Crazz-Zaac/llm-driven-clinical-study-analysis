"use client"

import { Header } from "@/components/header"
import { ChatInterface } from "@/components/chat-interface"
import { ContextSidebar } from "@/components/context-sidebar"

export default function Home() {
  return (
    <div className="flex flex-col h-screen bg-background">
      <Header />
      <main className="flex flex-1 overflow-hidden">
        <ContextSidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatInterface />
        </div>
      </main>
    </div>
  )
}
