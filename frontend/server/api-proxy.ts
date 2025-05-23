import type { Express, Request, Response } from "express";
import fetch from 'node-fetch';

const API_BASE_URL = 'https://5d15-2405-6e00-28eb-46f3-50ee-6733-6bc4-9696.ngrok-free.app';

export function setupApiProxy(app: Express) {
  // Proxy API routes to the actual backend API
  app.use('/api', async (req: Request, res: Response) => {
    try {
      // Map the routes to match the external API endpoints
      let apiPath = req.url;
      
      // Log the incoming request for debugging
      console.log(`API Proxy: ${req.method} ${req.url}`);
      
      const targetUrl = `${API_BASE_URL}${apiPath}`;
      const method = req.method;
      
      // Forward headers including authorization
      const headers: Record<string, string> = {};
      
      // Copy specific headers from the original request
      if (req.headers.authorization) {
        headers['Authorization'] = req.headers.authorization as string;
      }
      
      if (req.headers['content-type']) {
        headers['Content-Type'] = req.headers['content-type'] as string;
      }

      // Setup fetch options
      const fetchOptions: {
        method: string;
        headers: Record<string, string>;
        body?: string;
      } = {
        method,
        headers
      };
      
      // Only include body for methods that support it
      if (method !== 'GET' && method !== 'HEAD' && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }

      // Forward the request to the API
      const response = await fetch(targetUrl, fetchOptions);
      
      // Set the status code from the API response
      res.status(response.status);
      
      // Forward response headers
      response.headers.forEach((value, key) => {
        res.setHeader(key, value);
      });
      
      // Parse and forward the response body
      const data = await response.text();
      
      try {
        // Try to parse as JSON
        const jsonData = JSON.parse(data);
        res.json(jsonData);
      } catch (e) {
        // If not JSON, send as text
        res.send(data);
      }
      
    } catch (error) {
      console.error('API Proxy Error:', error);
      // Provide more detailed error information in development
      const errorMessage = process.env.NODE_ENV === 'development' 
        ? `Internal server error in API proxy: ${error instanceof Error ? error.message : 'Unknown error'}` 
        : 'Internal server error in API proxy';
      
      res.status(500).json({ error: errorMessage });
    }
  });
}
