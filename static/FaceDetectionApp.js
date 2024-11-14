import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';

const FaceDetectionApp = () => {
  const [isVideoActive, setIsVideoActive] = useState(false);
  const [showNewFaceAlert, setShowNewFaceAlert] = useState(false);

  // Function to handle starting the video
  const startVideo = () => {
    setIsVideoActive(true);
    // Start listening for new face events
    setupNewFaceEventSource();
  };

  // Function to handle stopping the video
  const stopVideo = () => {
    setIsVideoActive(false);
    // Clean up event source
    if (window.faceEventSource) {
      window.faceEventSource.close();
    }
  };

  // Setup event source for server-sent events
  const setupNewFaceEventSource = () => {
    if (window.faceEventSource) {
      window.faceEventSource.close();
    }
    
    const eventSource = new EventSource('/face-events');
    eventSource.onmessage = (event) => {
      if (event.data === 'new_face') {
        setShowNewFaceAlert(true);
        // Hide alert after 5 seconds
        setTimeout(() => setShowNewFaceAlert(false), 5000);
      }
    };
    
    window.faceEventSource = eventSource;
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (window.faceEventSource) {
        window.faceEventSource.close();
      }
    };
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-center mb-8">
        Real-Time Emotion Detection with Face Similarity
      </h1>

      <div className="flex gap-4 justify-center mb-6">
        <button
          onClick={startVideo}
          disabled={isVideoActive}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
        >
          Start Video
        </button>
        <button
          onClick={stopVideo}
          disabled={!isVideoActive}
          className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
        >
          Stop Video
        </button>
      </div>

      {showNewFaceAlert && (
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>New face detected!</AlertTitle>
          <AlertDescription>
            The system has detected a new face and added it to the database.
          </AlertDescription>
        </Alert>
      )}

      {isVideoActive && (
        <div className="rounded-lg overflow-hidden border border-gray-200">
          <img
            src="/video_feed"
            alt="Video feed"
            className="w-full"
          />
        </div>
      )}
    </div>
  );
};

export default FaceDetectionApp;