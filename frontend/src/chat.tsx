import React, { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send } from "lucide-react"
import { ChatMessage, ChatEntry, ChatState } from "@/types"
import { cn } from "@/lib/utils"
import { format } from 'date-fns'

const EconomicChat: React.FC = () => {
    const [query, setQuery] = useState('')
    const [chatState, setChatState] = useState<ChatState>({
        entries: [],
        loading: false
    })
    const scrollAreaRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (scrollAreaRef.current) {
            scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
        }
    }, [chatState.entries])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!query.trim() || chatState.loading) return

        const newEntry: ChatEntry = {
            id: crypto.randomUUID(),
            query,
            messages: [],
            timestamp: new Date()
        }

        setChatState(prev => ({
            ...prev,
            loading: true,
            entries: [...prev.entries, newEntry]
        }))

        try {
            const response = await fetch('http://localhost:8000/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            })

            const reader = response.body!.getReader()
            const decoder = new TextDecoder()

            while (true) {
                const { value, done } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value)
                const lines = chunk.split('\n').filter(line => line.trim())

                for (const line of lines) {
                    const message = JSON.parse(line) as ChatMessage

                    setChatState(prev => {
                        const currentEntry = {...prev.entries[prev.entries.length - 1]}

                        switch (message.type) {
                            case 'visualization':
                                currentEntry.plotUrl = `http://localhost:8000${message.content.plot_url}`
                                break
                            case 'source':
                                currentEntry.sourceInfo = message.content
                                break
                            case 'text':
                                currentEntry.messages = [...currentEntry.messages, message.content]
                                break
                            case 'error':
                                currentEntry.messages = [...currentEntry.messages, `Error: ${message.content.message}`]
                                break
                        }

                        const updatedEntries = [...prev.entries]
                        updatedEntries[updatedEntries.length - 1] = currentEntry

                        return {
                            ...prev,
                            entries: updatedEntries
                        }
                    })
                }
            }
        } catch (error) {
            console.error('Error:', error)
            setChatState(prev => {
                const currentEntry = {...prev.entries[prev.entries.length - 1]}
                currentEntry.messages = [...currentEntry.messages, 'An error occurred while processing your request.']

                const updatedEntries = [...prev.entries]
                updatedEntries[updatedEntries.length - 1] = currentEntry

                return {
                    ...prev,
                    entries: updatedEntries
                }
            })
        } finally {
            setChatState(prev => ({
                ...prev,
                loading: false
            }))
            setQuery('')
        }
    }

    const ChatMessage: React.FC<{ entry: ChatEntry }> = ({ entry }) => (
        <div className="mb-8 last:mb-0">
            {/* Query */}
            <div className="flex items-start mb-4">
                <div className="bg-primary text-primary-foreground rounded-lg py-2 px-4 max-w-[80%]">
                    <p className="text-sm">{entry.query}</p>
                    <p className="text-xs opacity-70 mt-1">
                        {format(entry.timestamp, 'HH:mm')}
                    </p>
                </div>
            </div>

            {/* Response */}
            <div className="flex flex-col items-start space-y-4">
                <div className="bg-muted rounded-lg py-3 px-6 max-w-[100%]">
                    {/* Text messages */}
                    <div className="space-y-1">
                        {entry.messages.map((message, index) => (
                            <div
                                key={index}
                                className={cn(
                                    "text-sm font-mono leading-relaxed",
                                    message.startsWith('•') ? "pl-4" : "",
                                    !message.trim() ? "h-4" : ""
                                )}
                            >
                                {message}
                            </div>
                        ))}
                    </div>

                    {/* Plot */}
                    {entry.plotUrl && (
                        <div className="mt-4">
                            <img
                                src={entry.plotUrl}
                                alt="Economic Data Visualization"
                                className="max-w-full rounded-lg shadow-lg"
                            />
                        </div>
                    )}

                    {/* Source Information */}
                    {entry.sourceInfo && (
                        <div className="mt-4 p-3 bg-background rounded-lg text-xs font-mono whitespace-pre-line">
                            {entry.sourceInfo}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )

    return (
        <div className="container mx-auto p-14 max-w-full">
            <Card className="w-full">
                <CardHeader>
                    <CardTitle className="text-center">Macro Specialist</CardTitle>
                </CardHeader>

                <CardContent>
                    <ScrollArea
                        ref={scrollAreaRef}
                        className="h-[70vh] w-full rounded-md border p-4"
                    >
                        {chatState.entries.map((entry) => (
                            <ChatMessage key={entry.id} entry={entry} />
                        ))}
                    </ScrollArea>
                </CardContent>

                <CardFooter>
                    <form onSubmit={handleSubmit} className="w-full flex gap-2">
                        <Input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Ask about economic data..."
                            disabled={chatState.loading}
                            className="flex-1"
                        />
                        <Button type="submit" disabled={chatState.loading}>
                            {chatState.loading ? (
                                <div className="flex items-center gap-2">
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    Processing
                                </div>
                            ) : (
                                <div className="flex items-center gap-2">
                                    <Send className="w-4 h-4" />
                                    Send
                                </div>
                            )}
                        </Button>
                    </form>
                </CardFooter>
            </Card>
        </div>
    )
}

export default EconomicChat