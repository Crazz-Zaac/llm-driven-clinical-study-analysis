"use client"

import { useAppStore } from "@/lib/store"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    FileText,
    Link as LinkIcon,
    Settings2,
    Database,
    Zap,
    ChevronDown,
} from "lucide-react"
import { useState } from "react"
import { cn } from "@/lib/utils"

interface SidebarSection {
    title: string
    icon: React.ReactNode
    isExpanded: boolean
}

export function ContextSidebar() {
    const {
        contextSources,
        selectedChatModel,
        useHostedModels,
        hostedModelName,
        temperature,
        maxTokens,
    } = useAppStore()

    const [expandedSections, setExpandedSections] = useState({
        documents: true,
        model: true,
        parameters: true,
    })

    const toggleSection = (section: keyof typeof expandedSections) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section],
        }))
    }

    const pdfSources = contextSources.filter(s => s.type === "pdf")
    const urlSources = contextSources.filter(s => s.type === "url")
    const activeModelName = useHostedModels ? hostedModelName : selectedChatModel

    const Section = ({
        id,
        title,
        icon,
        children,
        isEmpty,
    }: {
        id: keyof typeof expandedSections
        title: string
        icon: React.ReactNode
        children: React.ReactNode
        isEmpty?: boolean
    }) => (
        <div className="border-b border-border last:border-b-0">
            <button
                onClick={() => toggleSection(id)}
                className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <div className="text-muted-foreground">{icon}</div>
                    <span className="font-medium text-sm">{title}</span>
                    {!isEmpty && (
                        <Badge variant="secondary" className="text-xs ml-auto mr-2">
                            {id === "documents"
                                ? contextSources.length
                                : id === "model"
                                    ? activeModelName
                                        ? "Active"
                                        : "Inactive"
                                    : ""}
                        </Badge>
                    )}
                </div>
                <ChevronDown
                    className={cn(
                        "h-4 w-4 text-muted-foreground transition-transform",
                        expandedSections[id] ? "rotate-180" : ""
                    )}
                />
            </button>
            {expandedSections[id] && (
                <div className="px-3 py-2 bg-muted/20 border-t border-border/50">
                    {children}
                </div>
            )}
        </div>
    )

    return (
        <div className="w-64 border-r border-border bg-background/50 flex flex-col h-full">
            <div className="p-4 border-b border-border shrink-0">
                <h2 className="font-semibold text-sm">Context</h2>
                <p className="text-xs text-muted-foreground mt-1">
                    Chat configuration & sources
                </p>
            </div>

            <div className="flex-1 min-h-0 overflow-hidden">
                <ScrollArea className="h-full">
                    <div className="divide-y divide-border">
                        {/* Documents Section */}
                        <Section
                            id="documents"
                            title="Documents"
                            icon={<FileText className="h-4 w-4" />}
                            isEmpty={contextSources.length === 0}
                        >
                            {contextSources.length === 0 ? (
                                <div className="py-2 text-xs text-muted-foreground">
                                    <p>No documents added yet.</p>
                                    <p className="mt-1">
                                        Upload PDFs or add URLs in Settings → Context.
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {pdfSources.length > 0 && (
                                        <div>
                                            <p className="text-xs font-medium text-muted-foreground mb-1">
                                                PDFs ({pdfSources.length})
                                            </p>
                                            <div className="space-y-1">
                                                {pdfSources.slice(0, 3).map(source => (
                                                    <div
                                                        key={source.id}
                                                        className="text-xs p-1.5 rounded bg-muted/50 truncate hover:bg-muted transition-colors"
                                                        title={source.name}
                                                    >
                                                        {source.name}
                                                    </div>
                                                ))}
                                                {pdfSources.length > 3 && (
                                                    <div className="text-xs p-1.5 text-muted-foreground italic">
                                                        +{pdfSources.length - 3} more
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {urlSources.length > 0 && (
                                        <div>
                                            <p className="text-xs font-medium text-muted-foreground mb-1">
                                                URLs ({urlSources.length})
                                            </p>
                                            <div className="space-y-1">
                                                {urlSources.slice(0, 3).map(source => (
                                                    <div
                                                        key={source.id}
                                                        className="text-xs p-1.5 rounded bg-muted/50 truncate hover:bg-muted transition-colors flex items-center gap-1"
                                                        title={source.name}
                                                    >
                                                        <LinkIcon className="h-3 w-3 shrink-0 text-muted-foreground" />
                                                        {source.name}
                                                    </div>
                                                ))}
                                                {urlSources.length > 3 && (
                                                    <div className="text-xs p-1.5 text-muted-foreground italic">
                                                        +{urlSources.length - 3} more
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </Section>

                        {/* Model Section */}
                        <Section
                            id="model"
                            title="Model"
                            icon={<Database className="h-4 w-4" />}
                            isEmpty={!activeModelName}
                        >
                            {!activeModelName ? (
                                <div className="py-2 text-xs text-muted-foreground">
                                    <p>No model selected.</p>
                                    <p className="mt-1">
                                        Configure a model in Settings → Models.
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <div className="text-xs">
                                        <p className="text-muted-foreground mb-1">Selected Chat Model</p>
                                        <div className="p-2 rounded bg-muted/50 font-mono text-xs break-words">
                                            {activeModelName}
                                        </div>
                                    </div>
                                    <div className="text-xs">
                                        <p className="text-muted-foreground mb-1">Type</p>
                                        <Badge variant="outline" className="text-xs">
                                            {useHostedModels ? "Hosted" : "Local"}
                                        </Badge>
                                    </div>
                                </div>
                            )}
                        </Section>

                        {/* Parameters Section */}
                        <Section
                            id="parameters"
                            title="Parameters"
                            icon={<Zap className="h-4 w-4" />}
                        >
                            <div className="space-y-3">
                                <div className="text-xs">
                                    <p className="text-muted-foreground mb-1">Temperature</p>
                                    <div className="p-2 rounded bg-muted/50">
                                        <div className="font-mono">
                                            {temperature?.toFixed(2) || "0.70"}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            {(temperature || 0.7) < 0.5
                                                ? "Deterministic"
                                                : (temperature || 0.7) > 1
                                                    ? "Creative"
                                                    : "Balanced"}
                                        </p>
                                    </div>
                                </div>

                                <div className="text-xs">
                                    <p className="text-muted-foreground mb-1">Max Tokens</p>
                                    <div className="p-2 rounded bg-muted/50">
                                        <div className="font-mono">
                                            {maxTokens || "512"}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Max response length
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </Section>
                    </div>
                </ScrollArea>
            </div>

            <div className="p-3 border-t border-border text-xs text-muted-foreground shrink-0">
                <p className="flex items-center gap-1">
                    <Settings2 className="h-3 w-3" />
                    Configure in Settings
                </p>
            </div>
        </div>
    )
}
