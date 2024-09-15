"use client";
import { useCallback, useRef, useState, useEffect } from "react";
import { ForceGraph3D } from "react-force-graph";
import { AudioLines } from "lucide-react";
import { data } from "../data";
import "../styles/app.scss";
import { endpoints } from "../config";
import axios from "axios";
import { sha256 } from "crypto-js";
import SpriteText from "three-spritetext";
import {
  ChartComponent,
} from "./Charts";
import { temp_data } from "./data";

export default function ForceGraph3DComponent() {
  const [isMounted, setIsMounted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const fgRef = useRef();
  const [focusedNodeId, setFocusedNodeId] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [liveTranscription, setLiveTranscription] = useState("");
  const [finalTranscription, setFinalTranscription] = useState("");
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recognitionRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const gainNodeRef = useRef(null);
  const audioBufferRef = useRef([]);
  const audioCache = useRef(new Map());
  const [context, setContext] = useState("");
  const textareaRef = useRef(null);
  const [wsConnection, setWsConnection] = useState(null);
  const audioQueue = useRef([]);
  const [contextCharts, setContextCharts] = useState(null)
  const [activeContextNodeIDs, setActiveContextNodeIDs] = useState(null);
  const isPlayingAudio = useRef(false);
  const [contextText, setContextText] = useState("");
  useEffect(() => {
    audioContextRef.current = new (window.AudioContext ||
      window.webkitAudioContext)();
    gainNodeRef.current = audioContextRef.current.createGain();
    gainNodeRef.current.connect(audioContextRef.current.destination);

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new WebSocket("ws://localhost:8555/ws");
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("WebSocket connection established");
      setWsConnection(ws);
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        try {
          const jsonData = JSON.parse(event.data);
          if (jsonData.type === "node_ids") {
            setActiveContextNodeIDs(jsonData.data);

            handleSend(jsonData.data[0]);
          }
          else if (jsonData.type === "chart") {
            setContextCharts(jsonData.data);
       
          }
          else if (jsonData.type === "reply") {
            setContextText(prevText => prevText + jsonData.data.text);
          }
        } catch (error) {
          console.error("Error parsing JSON:", error);
        }
      } else if (event.data instanceof ArrayBuffer) {
        audioQueue.current.push(event.data);
        if (!isPlayingAudio.current) {
          playNextAudioChunk();
        }
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const handleSend = useCallback(
    (jsonData) => {
      const node = data.nodes.find(
        (n) => n.id === jsonData || n.id.toString() === jsonData
      );

      if (node) {
        const distance = 500;
        const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

        fgRef.current.cameraPosition(
          {
            x: node.x * distRatio,
            y: node.y * distRatio,
            z: node.z * distRatio,
          },
          node,
          3000
        );
        setFocusedNodeId(node.id);
      } else {
        alert("Node not found");
      }
    },
    [activeContextNodeIDs, data, fgRef]
  );

  const handleClick = useCallback(
    (node) => {
      // Aim at node from outside it
      const distance = 200;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

      fgRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
        node, // lookAt ({ x, y, z })
        3000 // ms transition duration
      );
    },
    [fgRef]
  );

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        setAudioBlob(audioBlob);
        sendAudioToBackend(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setFinalTranscription("");
      startSpeechRecognition();
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      stopSpeechRecognition();
    }
  };

  const startSpeechRecognition = () => {
    recognitionRef.current = new (window.SpeechRecognition ||
      window.webkitSpeechRecognition)();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(" ");
      setLiveTranscription(transcript);
    };

    recognitionRef.current.start();
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav"); // Change 'audio' to 'file'

      const response = await axios.post(endpoints.transcribe_audio, formData);

      setFinalTranscription(response.data.transcription);
      setContext(
        (prevContext) => prevContext + " " + response.data.transcription
      );
    } catch (error) {
      console.error("Error sending audio to backend:", error);
    } finally {
      setIsProcessing(false);
      setLiveTranscription("");
    }
  };

  const generateAudio = useCallback(async () => {
    if (!finalTranscription.trim() || isGeneratingAudio || !wsConnection)
      return;
    setActiveContextNodeIDs([]);
    setIsGeneratingAudio(true);
    audioQueue.current = [];

    try {
      wsConnection.send(finalTranscription);
    } catch (error) {
      console.error("Error sending message to WebSocket:", error);
    } finally {
      setIsGeneratingAudio(false);
    }
  }, [finalTranscription, isGeneratingAudio, wsConnection]);

  const playNextAudioChunk = async () => {
    if (audioQueue.current.length === 0) {
      isPlayingAudio.current = false;
      return;
    }

    isPlayingAudio.current = true;
    const audioData = audioQueue.current.shift();
    const audioBuffer = await audioContextRef.current.decodeAudioData(
      audioData
    );

    const sourceNode = audioContextRef.current.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(gainNodeRef.current);
    sourceNode.start();
    sourceNode.onended = playNextAudioChunk;
  };

  const playAudioStream = useCallback(() => {
    const playNextChunk = async () => {
      if (audioBufferRef.current.length === 0) {
        return;
      }

      const sourceNode = audioContextRef.current.createBufferSource();
      sourceNode.connect(gainNodeRef.current);

      const audioData = audioBufferRef.current.shift();
      const audioBuffer = await audioContextRef.current.decodeAudioData(
        audioData.buffer
      );

      sourceNode.buffer = audioBuffer;
      sourceNode.start();
      sourceNode.onended = playNextChunk;
    };

    playNextChunk();
  }, []);

  // Updated function with error handling and fallback
  const setInitialCameraPosition = useCallback(() => {
    if (fgRef.current) {
      const distance = 1; // Adjust this value to change how close the camera is
      const { x, y, z } = fgRef.current.cameraPosition();

      fgRef.current.cameraPosition(
        { x: x * distance, y: y * distance, z: z * distance }, // new position
        { x: 10, y: 0, z: 0 }, // lookAt center of the scene
        1 // set duration to 0 for immediate effect
      );
    }
  }, []);

  // Use useEffect to set the initial camera position when the component mounts
  useEffect(() => {
    setInitialCameraPosition();
  }, [setInitialCameraPosition]);

  // Add this new effect
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [liveTranscription, finalTranscription]);

  return (
    <div className="app">
      <div className="chat">
        <div className="content">
          <div className="main">
            <p>
                {contextText}
            </p>
            {contextCharts && contextCharts.map((xa, index) => {
                let propc  ={}
                if(xa.type =='bar'){
                    propc['xAxisType']='band'
                }
              return (
                <div className="chart">
                 
                  <ChartComponent
                    data={{
                      ...xa.data,
                      ...propc,
                      chartType: xa.type,
                    }}
                  />
                     <label htmlFor="">
                        {xa.title}
                    </label>
                  ;
                </div>
              );
            })}
          </div>
          <div className="bottom">
            <div className="input-box">
              <div
                className={`left ${
                  isRecording || isProcessing ? "blurred" : ""
                }`}
              >
                {isRecording ? (
                  <div className="reddot" onClick={stopRecording}></div>
                ) : (
                  <AudioLines onClick={startRecording} />
                )}
                <textarea
                  ref={textareaRef}
                  placeholder={
                    isRecording ? "Recording..." : "Enter node ID or speak..."
                  }
                  value={finalTranscription || liveTranscription || inputValue}
                  onChange={(e) => {
                    setInputValue(e.target.value);
                    setFinalTranscription(e.target.value);

                    // Auto-resize logic
                    e.target.style.height = "auto";
                    e.target.style.height = `${e.target.scrollHeight}px`;
                  }}
                  className={isRecording || isProcessing ? "blurred" : ""}
                  rows={1}
                  style={{ overflow: "hidden", resize: "none" }}
                />
              </div>
              <button
                type="submit"
                onClick={isGeneratingAudio ? null : generateAudio}
              >
                {isGeneratingAudio ? "Talking to AI..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="force-graph">
        <ForceGraph3D
          ref={fgRef}
          className="force-graph"
          onBeforeRender={() => setIsMounted(true)}
          graphData={data}
          nodeThreeObject={(node) => {
            if (activeContextNodeIDs?.includes(node.id)) {
              const sprite = new SpriteText(
                node.title || node.name || node.id.toString()
              );
              sprite.color = "white";
              sprite.textHeight = 8;
              sprite.visible =
                activeContextNodeIDs && activeContextNodeIDs.includes(node.id);
              return sprite;
            }
          }}
          nodeAutoColorBy="type"
          onNodeClick={handleClick}
          linkLabel={(link) =>
            `${link.source.title || link.source.name} - ${
              link.target.title || link.target.name
            }`
          }
          nodeColor={(node) => {
            if (activeContextNodeIDs) {
              if (activeContextNodeIDs.includes(node.id)) {
                return "yellow"; // Highlight active context nodes
              } else {
                return "grey"; // Default node color
              }
            } else {
              switch (node.type) {
                case "company":
                  return "red";
                case "growth_analysis":
                  return "lightgreen";
                case "market_sentiment":
                  return "lightblue";
                case "advanced_metrics":
                  return "cyan";
                case "financial_insights":
                  return "white";
                case "financial_health":
                  return "pink";
                case "news":
                  return "white";

                default:
                  return "white";
              }
            }
          }}
          linkWidth={focusedNodeId ? 0.2 : 0.6}
        />
      </div>
    </div>
  );
}
