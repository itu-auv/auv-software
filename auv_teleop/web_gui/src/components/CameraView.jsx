import React, { useState, useEffect } from 'react';
import { RefreshCw, Video, Camera, AlertCircle, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';

function CameraView({ connected, ros }) {
  const [selectedTopics, setSelectedTopics] = useState({
    camera1: '',
    camera2: '',
    camera3: ''
  });
  const [imageTopics, setImageTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [streamStatus, setStreamStatus] = useState({
    camera1: 'loading',
    camera2: 'loading',
    camera3: 'loading'
  });

  const discoverImageTopics = () => {
    if (!ros) return;

    setLoading(true);
    ros.getTopics((result) => {
      const imgTopics = result.topics.filter((topic, index) => {
        const type = result.types[index];
        return type === 'sensor_msgs/Image' ||
               type === 'sensor_msgs/CompressedImage' ||
               topic.includes('image') ||
               topic.includes('camera');
      });

      setImageTopics(imgTopics);

      if (imgTopics.length > 0 && !selectedTopics.camera1) {
        const frontCam = imgTopics.find(t => t.includes('front') || t.includes('cam_front'));
        const bottomCam = imgTopics.find(t => t.includes('bottom') || t.includes('cam_bottom'));
        const detectionCam = imgTopics.find(t => t.includes('detection') || t.includes('result') || t.includes('annotated'));

        setSelectedTopics({
          camera1: frontCam || imgTopics[0] || '',
          camera2: bottomCam || (imgTopics.length > 1 ? imgTopics[1] : '') || '',
          camera3: detectionCam || (imgTopics.length > 2 ? imgTopics[2] : '') || ''
        });
      }

      setLoading(false);
    }, () => setLoading(false));
  };

  useEffect(() => {
    if (connected && ros) {
      discoverImageTopics();
    }
  }, [connected, ros]);

  const getStreamUrl = (topic) => {
    if (!topic) return null;
    const hostname = window.location.hostname || 'localhost';
    return `http://${hostname}:8080/stream?topic=${topic}&type=mjpeg&quality=80`;
  };

  const handleTopicChange = (camera, topic) => {
    setSelectedTopics(prev => ({ ...prev, [camera]: topic }));
    setStreamStatus(prev => ({ ...prev, [camera]: 'loading' }));
  };

  const handleImageLoad = (camera) => {
    setStreamStatus(prev => ({ ...prev, [camera]: 'connected' }));
  };

  const handleImageError = (camera) => {
    setStreamStatus(prev => ({ ...prev, [camera]: 'error' }));
  };

  if (!connected) {
    return (
      <div className="w-full">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="w-5 h-5 text-white/50" />
              Camera View
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Alert variant="warning">
              <AlertCircle className="w-4 h-4" />
              <AlertDescription>
                Not connected to ROS. Please connect to view camera streams.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>
    );
  }

  const CameraPanel = ({ cameraKey, label }) => {
    const topic = selectedTopics[cameraKey];
    const status = streamStatus[cameraKey];

    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Camera className="w-4 h-4 text-white/50" />
              {label}
            </CardTitle>
            {topic && (
              <Badge
                variant={status === 'connected' ? 'success' : status === 'error' ? 'destructive' : 'secondary'}
                className="text-xs"
              >
                {status === 'connected' ? 'Live' : status === 'error' ? 'Error' : 'Loading...'}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Select
            value={topic || "none"}
            onValueChange={(v) => handleTopicChange(cameraKey, v === "none" ? "" : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select camera topic" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              {imageTopics.map((t) => (
                <SelectItem key={t} value={t}>
                  {t.split('/').slice(-2).join('/')}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="relative w-full aspect-video bg-black/50 rounded-lg overflow-hidden border border-white/10">
            {topic ? (
              <>
                {status === 'loading' && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin text-white/50 mx-auto mb-2" />
                      <p className="text-xs text-white/50">Connecting to stream...</p>
                    </div>
                  </div>
                )}
                {status === 'error' && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                    <div className="text-center p-4">
                      <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
                      <p className="text-sm text-red-400 font-medium">Stream unavailable</p>
                      <p className="text-xs text-white/40 mt-1">Make sure web_video_server is running:</p>
                      <code className="text-xs text-white/60 bg-white/5 px-2 py-1 rounded mt-2 block">
                        rosrun web_video_server web_video_server
                      </code>
                    </div>
                  </div>
                )}
                <img
                  key={topic}
                  src={getStreamUrl(topic)}
                  alt={label}
                  className="w-full h-full object-contain"
                  onLoad={() => handleImageLoad(cameraKey)}
                  onError={() => handleImageError(cameraKey)}
                />
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="w-10 h-10 text-white/20 mx-auto mb-2" />
                  <p className="text-sm text-white/30">No camera selected</p>
                </div>
              </div>
            )}
          </div>

          {topic && (
            <p className="text-xs text-white/30 truncate" title={topic}>
              Topic: {topic}
            </p>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Video className="w-5 h-5 text-white/50" />
          <h2 className="text-lg font-semibold">Camera Streams</h2>
          <Badge variant={imageTopics.length > 0 ? "success" : "secondary"}>
            {imageTopics.length} topic{imageTopics.length !== 1 ? 's' : ''} found
          </Badge>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={discoverImageTopics}
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {loading ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin text-white/50 mx-auto mb-3" />
              <p className="text-white/50">Discovering camera topics...</p>
            </div>
          </CardContent>
        </Card>
      ) : imageTopics.length === 0 ? (
        <Alert variant="warning">
          <AlertCircle className="w-4 h-4" />
          <AlertDescription>
            <strong>No camera topics found!</strong>
            <p className="mt-2 text-sm text-white/60">
              Make sure camera nodes and web_video_server are running:
            </p>
            <code className="block text-xs mt-2 bg-black/30 p-2 rounded font-mono">
              rosrun web_video_server web_video_server _port:=8080
            </code>
          </AlertDescription>
        </Alert>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          <CameraPanel cameraKey="camera1" label="Camera 1 (Front)" />
          <CameraPanel cameraKey="camera2" label="Camera 2 (Bottom)" />
          <CameraPanel cameraKey="camera3" label="Camera 3" />
        </div>
      )}

      {/* Stream Server Info */}
      {imageTopics.length > 0 && (
        <Card className="bg-white/[0.02]">
          <CardContent className="py-3">
            <div className="flex items-center justify-between text-xs text-white/40">
              <span>Stream Server: http://{window.location.hostname || 'localhost'}:8080</span>
              <span>{imageTopics.length} available topics</span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default CameraView;
