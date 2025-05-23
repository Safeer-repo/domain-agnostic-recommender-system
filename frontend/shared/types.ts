export interface DomainPreferences {
  entertainment: string[];
  ecommerce: string[];
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  created_at: number;
  last_updated: number;
  domain_preferences: DomainPreferences;
  metadata?: Record<string, any>;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
}

export interface EntertainmentRecommendation {
  id: string;
  title: string;
  description: string;
  image?: string;
  year: string;
  genre: string;
  rating: number;
}

export interface EcommerceRecommendation {
  id: string;
  title: string;
  description: string;
  image?: string;
  price: string;
  category: string;
  rating: number;
}

export type Recommendation = EntertainmentRecommendation | EcommerceRecommendation;
