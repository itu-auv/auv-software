import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Grid, 
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  IconButton,
  Chip
} from '@mui/material';
import { Refresh, Videocam } from '@mui/icons-material';

function CameraView({ connected, ros }) {
  const [availableTopics, setAvailableTopics] = useState([]);
  const [selectedTopics, setSelectedTopics] = useState({
    camera1: '',
    camera2: ''
  });
  const [imageTopics, setImageTopics] = useState([]);
  const [loading, setLoading] = useState(true);

  // Discover image topics
  const discoverImageTopics = () => {
    if (!ros) return;
    
    setLoading(true);
    ros.getTopics((result) => {
      console.log('📋 All topics:', result.topics);
      console.log('📋 Topic types:', result.types);
      
      // Filter for image topics (sensor_msgs/Image or sensor_msgs/CompressedImage)
      const imgTopics = result.topics.filter((topic, index) => {
        const type = result.types[index];
        return type === 'sensor_msgs/Image' || 
               type === 'sensor_msgs/CompressedImage' ||
               topic.includes('image') ||
               topic.includes('camera');
      });
      
      console.log('📷 Found image topics:', imgTopics);
      setImageTopics(imgTopics);
      
      // Auto-select common camera topics if available
      if (imgTopics.length > 0 && !selectedTopics.camera1) {
        const frontCam = imgTopics.find(t => t.includes('front') || t.includes('cam_front'));
        const bottomCam = imgTopics.find(t => t.includes('bottom') || t.includes('cam_bottom'));
        
        setSelectedTopics({
          camera1: frontCam || imgTopics[0] || '',
          camera2: bottomCam || (imgTopics.length > 1 ? imgTopics[1] : '') || ''
        });
      }
      
      setLoading(false);
    }, (error) => {
      console.error('❌ Failed to get topics:', error);
      setLoading(false);
    });
  };

  useEffect(() => {
    if (connected && ros) {
      discoverImageTopics();
    }
  }, [connected, ros]);

  // Get MJPEG stream URL for a topic
  const getStreamUrl = (topic) => {
    if (!topic) return null;
    
    // Use web_video_server if available (port 8080 is default)
    // Format: http://localhost:8080/stream?topic=/camera/image_raw
    const hostname = window.location.hostname || 'localhost';
    return `http://${hostname}:8080/stream?topic=${topic}&type=mjpeg`;
  };

  const handleTopicChange = (camera, topic) => {
    setSelectedTopics(prev => ({
      ...prev,
      [camera]: topic
    }));
  };

  if (!connected) {
    return (
      <Box sx={{ width: '100%', px: 3 }}>
        <Typography variant="h6" mb={2}>📹 Camera View</Typography>
        <Alert severity="warning">
          Not connected to ROS. Please connect to view camera streams.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', px: 3 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2} sx={{ width: '100%' }}>
          <Box display="flex" alignItems="center" gap={1}>
            <Videocam fontSize="large" color="primary" />
            <Typography variant="h6">Camera View</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={`${imageTopics.length} camera${imageTopics.length !== 1 ? 's' : ''} found`}
              size="small"
              color={imageTopics.length > 0 ? "success" : "default"}
            />
            <IconButton 
              size="small" 
              onClick={discoverImageTopics}
              disabled={loading}
            >
              <Refresh />
            </IconButton>
          </Box>
        </Box>

        {loading ? (
          <Alert severity="info" sx={{ width: '100%' }}>Discovering camera topics...</Alert>
        ) : imageTopics.length === 0 ? (
          <Box sx={{ width: '100%' }}>
          <Alert severity="warning" sx={{ width: '100%' }}>
            <Typography variant="body2" gutterBottom>
              <strong>No camera topics found!</strong>
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Make sure camera nodes are running and web_video_server is started:
            </Typography>
            <code style={{ display: 'block', fontSize: '10px', marginTop: 4 }}>
              # Start web_video_server:<br/>
              rosrun web_video_server web_video_server<br/>
              <br/>
              # Or with custom port:<br/>
              rosrun web_video_server web_video_server _port:=8080
            </code>
          </Alert>
          </Box>
        ) : (
          <Grid container spacing={2} sx={{ width: '100%', margin: 0 }}>
            {/* Camera 1 */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Camera 1</InputLabel>
                <Select
                  value={selectedTopics.camera1}
                  label="Camera 1"
                  onChange={(e) => handleTopicChange('camera1', e.target.value)}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  {imageTopics.map((topic) => (
                    <MenuItem key={topic} value={topic}>
                      {topic}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedTopics.camera1 ? (
                <Box 
                  sx={{ 
                    width: '100%',
                    height: 300,
                    bgcolor: 'black',
                    borderRadius: 1,
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <img 
                    src={getStreamUrl(selectedTopics.camera1)}
                    alt="Camera 1"
                    style={{ 
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain'
                    }}
                    onError={(e) => {
                      console.error('Failed to load camera stream:', selectedTopics.camera1);
                      e.target.style.display = 'none';
                      e.target.parentElement.innerHTML += '<div style="color: #ff5252; text-align: center;">Stream unavailable<br/><small>Check web_video_server</small></div>';
                    }}
                  />
                </Box>
              ) : (
                <Box 
                  sx={{ 
                    width: '100%',
                    height: 300,
                    bgcolor: 'background.paper',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px dashed rgba(255,255,255,0.2)'
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    No camera selected
                  </Typography>
                </Box>
              )}
            </Grid>

            {/* Camera 2 */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Camera 2</InputLabel>
                <Select
                  value={selectedTopics.camera2}
                  label="Camera 2"
                  onChange={(e) => handleTopicChange('camera2', e.target.value)}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  {imageTopics.map((topic) => (
                    <MenuItem key={topic} value={topic}>
                      {topic}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedTopics.camera2 ? (
                <Box 
                  sx={{ 
                    width: '100%',
                    height: 300,
                    bgcolor: 'black',
                    borderRadius: 1,
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <img 
                    src={getStreamUrl(selectedTopics.camera2)}
                    alt="Camera 2"
                    style={{ 
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain'
                    }}
                    onError={(e) => {
                      console.error('Failed to load camera stream:', selectedTopics.camera2);
                      e.target.style.display = 'none';
                      e.target.parentElement.innerHTML += '<div style="color: #ff5252; text-align: center;">Stream unavailable<br/><small>Check web_video_server</small></div>';
                    }}
                  />
                </Box>
              ) : (
                <Box 
                  sx={{ 
                    width: '100%',
                    height: 300,
                    bgcolor: 'background.paper',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px dashed rgba(255,255,255,0.2)'
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    No camera selected
                  </Typography>
                </Box>
              )}
            </Grid>
          </Grid>
        )}

        {/* Debug Info */}
        {imageTopics.length > 0 && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1, fontSize: '0.7rem' }}>
            <Typography variant="caption" display="block">
              <strong>Available Topics:</strong> {imageTopics.join(', ')}
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
              <strong>Stream Server:</strong> http://localhost:8080
            </Typography>
          </Box>
        )}
    </Box>
  );
}

export default CameraView;
