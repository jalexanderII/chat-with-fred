interface BaseMessage {
    type: 'text' | 'visualization' | 'error' | 'source';
}

interface TextMessage extends BaseMessage {
    type: 'text';
    content: string;
}

interface VisualizationMessage extends BaseMessage {
    type: 'visualization';
    content: {
        plot_url: string;
        format: string;
    };
}

interface ErrorMessage extends BaseMessage {
    type: 'error';
    content: {
        message: string;
        details?: string;
    };
}


interface SourceMessage extends BaseMessage {
    type: 'source';
    content: string;
}

export type ChatMessage = TextMessage | VisualizationMessage | ErrorMessage | SourceMessage;

export interface ChatEntry {
    id: string;
    query: string;
    messages: string[];
    plotUrl?: string;
    sourceInfo?: string;
    timestamp: Date;
}

export interface ChatState {
    entries: ChatEntry[];
    loading: boolean;
}