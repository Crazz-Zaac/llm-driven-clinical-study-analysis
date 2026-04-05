"use client"

import Link from "next/link"
import { useTheme } from "next-themes"
import { Settings, Moon, Sun, Cpu } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Header() {
  const { theme, setTheme } = useTheme()

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-14 items-center justify-between px-4 md:px-6">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
            <Cpu className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-semibold tracking-tight">LLM4EHR</span>
            <span className="hidden text-xs text-muted-foreground sm:block">AI-Powered Healthcare Research</span>
          </div>
        </Link>
        
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="h-9 w-9"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
          
          <Link href="/settings">
            <Button variant="outline" size="sm" className="gap-2">
              <Settings className="h-4 w-4" />
              <span className="hidden sm:inline">Settings</span>
            </Button>
          </Link>
        </div>
      </div>
    </header>
  )
}
