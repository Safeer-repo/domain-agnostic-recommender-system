import type { Express } from "express";
import { createServer, type Server } from "http";
import { setupApiProxy } from "./api-proxy";
import { storage } from "./storage";

export async function registerRoutes(app: Express): Promise<Server> {
  // Setup API proxy to forward requests to the recommendation API
  setupApiProxy(app);

  const httpServer = createServer(app);

  return httpServer;
}
