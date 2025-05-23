import { createContext, useContext, ReactNode, useState, useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useMutation } from "@tanstack/react-query";
import { User } from "@shared/types";
import { apiRequest, loginUser, registerUser, getUserProfile } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";
import { useLocation } from "wouter";

interface AuthContextProps {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  register: (username: string, email: string, password: string, domainPreference: 'entertainment' | 'ecommerce') => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextProps | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const { toast } = useToast();
  const [error, setError] = useState<string | null>(null);
  const [, setLocation] = useLocation();

  // Check if we have a token in local storage
  const hasToken = !!localStorage.getItem('token');

  // Fetch the user profile if we have a token
  const { 
    data: user, 
    isLoading,
    refetch
  } = useQuery<User | null>({
    // Use a consistent query key path that matches our API paths
    queryKey: ['/api/user/me'],
    queryFn: async () => {
      if (!hasToken) return null;
      
      const response = await getUserProfile();
      if (response.error) {
        if (response.error.status === 401) {
          // Clear token if unauthorized
          localStorage.removeItem('token');
          return null;
        }
        throw new Error(response.error.message);
      }
      return response.data as User;
    },
    enabled: hasToken,
    staleTime: 0, // Always fetch fresh data
    retry: 1,     // Only retry once to avoid excessive retries
  });

  // Login mutation
  const loginMutation = useMutation({
    mutationFn: async ({ username, password }: { username: string; password: string }) => {
      const response = await loginUser(username, password);
      if (response.error) {
        throw new Error(response.error.message);
      }
      return response.data;
    },
    onSuccess: async (data) => {
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user_id', data.user_id);
      
      // Immediately refetch user data and wait for it to complete
      await refetch();
      
      toast({
        title: "Login successful",
        description: "Welcome back!",
      });
      
      // Navigate to dashboard after successful login and data retrieval
      setLocation("/dashboard");
    },
    onError: (error: Error) => {
      toast({
        title: "Login failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Register mutation
  const registerMutation = useMutation({
    mutationFn: async ({ 
      username, 
      email, 
      password, 
      domainPreference 
    }: { 
      username: string; 
      email: string; 
      password: string; 
      domainPreference: 'entertainment' | 'ecommerce';
    }) => {
      const response = await registerUser(username, email, password, domainPreference);
      if (response.error) {
        throw new Error(response.error.message);
      }
      return response.data;
    },
    onSuccess: async (data) => {
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user_id', data.user_id);
      
      // Immediately refetch user data
      await refetch();
      
      toast({
        title: "Registration successful",
        description: "Your account has been created!",
      });
      
      // Navigate to dashboard after registration
      setLocation("/dashboard");
    },
    onError: (error: Error) => {
      toast({
        title: "Registration failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const login = async (username: string, password: string) => {
    try {
      await loginMutation.mutateAsync({ username, password });
      return true;
    } catch (err) {
      return false;
    }
  };

  const register = async (username: string, email: string, password: string, domainPreference: 'entertainment' | 'ecommerce') => {
    try {
      await registerMutation.mutateAsync({ username, email, password, domainPreference });
      return true;
    } catch (err) {
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user_id');
    // Update cache with the correct query key
    queryClient.setQueryData(['/api/user/me'], null);
    toast({
      title: "Logged out",
      description: "You have been logged out successfully.",
    });
    
    // Redirect to auth page after logout
    setLocation("/auth");
  };

  return (
    <AuthContext.Provider value={{ 
      user: user || null, 
      isLoading, 
      error, 
      login, 
      register, 
      logout 
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
