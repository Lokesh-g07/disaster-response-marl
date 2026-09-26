export class SimulationWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessageCallback: (data: any) => void;
  private onErrorCallback: (error: string) => void;
  private onCloseCallback: () => void;

  constructor(
    simulationId: string,
    onMessage: (data: any) => void,
    onError: (error: string) => void,
    onClose: () => void
  ) {
    const baseUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
    const wsUrl = baseUrl.replace(/^http/, 'ws') + `/ws/simulations/${simulationId}`;
    
    this.url = wsUrl;
    this.onMessageCallback = onMessage;
    this.onErrorCallback = onError;
    this.onCloseCallback = onClose;
  }

  public connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessageCallback(data);
      } catch (err) {
        console.error('Failed to parse WebSocket message', err);
      }
    };

    this.ws.onerror = () => {
      this.onErrorCallback('WebSocket connection error');
    };

    this.ws.onclose = () => {
      this.onCloseCallback();
      this.ws = null;
    };
  }

  public send(type: 'start' | 'pause' | 'step' | 'stop') {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type }));
    }
  }

  public disconnect() {
    if (this.ws) {
      // Avoid firing the close callback if we intentionally disconnect
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
      this.onCloseCallback();
    }
  }
}
