import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import axios from 'axios';

interface Message {
  role: 'assistant' | 'user';
  content: string;
}

const ChatSurvey = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [userId] = useState(localStorage.getItem('user_id'));
  const [surveyComplete, setSurveyComplete] = useState(false);

  useEffect(() => {
    // Initial message
    setMessages([
      {
        role: 'assistant',
        content: 'Welcome to the security survey. I will ask you some questions about your store security.',
      },
    ]);
    getNextQuestion();
  }, []);

  const getNextQuestion = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/survey/question');
      if (response.data.question) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: response.data.question,
        }]);
      } else {
        setSurveyComplete(true);
        // Get reports
        const quickReport = await axios.get(`http://localhost:5000/api/report/quick?user_id=${userId}`);
        const detailedReport = await axios.get(`http://localhost:5000/api/report/detailed?user_id=${userId}`);
        
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Survey complete! You can download your reports here:
          
          Quick Report: ${quickReport.data.report_url}
          Detailed Report: ${detailedReport.data.report_url}`,
        }]);
      }
    } catch (error) {
      console.error('Error getting next question:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: input,
    }]);

    try {
      // Save survey response
      await axios.post('http://localhost:5000/api/survey', {
        user_id: userId,
        question: messages[messages.length - 1].content,
        answer: input,
      });

      setInput('');
      getNextQuestion();
    } catch (error) {
      console.error('Error submitting answer:', error);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 3, minHeight: '70vh', display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h5" gutterBottom>
            Security Survey
          </Typography>
          
          <List sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
            {messages.map((message, index) => (
              <ListItem
                key={index}
                sx={{
                  justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                <Paper
                  elevation={1}
                  sx={{
                    p: 2,
                    maxWidth: '80%',
                    backgroundColor: message.role === 'user' ? '#e3f2fd' : '#f5f5f5',
                  }}
                >
                  <ListItemText primary={message.content} />
                </Paper>
              </ListItem>
            ))}
          </List>

          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your answer here..."
                variant="outlined"
                size="small"
              />
              <Button type="submit" variant="contained" color="primary">
                Send
              </Button>
            </Box>
          </form>
        </Paper>
      </Box>
    </Container>
  );
};

export default ChatSurvey;