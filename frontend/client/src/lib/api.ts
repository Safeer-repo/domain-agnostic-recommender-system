// Use our API proxy instead of direct external API calls
const API_BASE_URL = '/api';

export interface ApiResponse<T> {
  data?: T;
  error?: {
    status: number;
    message: string;
  };
}

export async function apiRequest<T = any>(
  endpoint: string,
  method: string = 'GET',
  data: any = null
): Promise<ApiResponse<T>> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  // Add authorization header if token exists
  const token = localStorage.getItem('token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const options: RequestInit = {
    method,
    headers,
    credentials: 'include',
  };

  if (data) {
    options.body = JSON.stringify(data);
  }

  try {
    console.log(`API Request: ${method} ${url}`);
    if (data) {
      console.log('Request payload:', data);
    }

    const response = await fetch(url, options);
    let responseData;
    
    try {
      responseData = await response.json();
    } catch (e) {
      responseData = null;
    }
    
    console.log('Response:', responseData);

    if (!response.ok) {
      return {
        error: {
          status: response.status,
          message: responseData?.detail || 'An error occurred',
        },
      };
    }

    return { data: responseData };
  } catch (error) {
    console.error('API request error:', error);
    return {
      error: {
        status: 500,
        message: 'Network or server error',
      },
    };
  }
}

// Login user
export async function loginUser(username: string, password: string) {
  return apiRequest('/user/login', 'POST', { username, password });
}

// Register user
export async function registerUser(
  username: string,
  email: string,
  password: string,
  domainPreference: 'entertainment' | 'ecommerce'
) {
  // Construct domain preferences based on selection
  let domain_preferences = {};
  if (domainPreference === 'entertainment') {
    domain_preferences = {
      entertainment: ["movielens"],
      ecommerce: []
    };
  } else {
    domain_preferences = {
      entertainment: [],
      ecommerce: ["amazon"]
    };
  }

  return apiRequest('/user/create', 'POST', {
    username,
    email,
    password,
    domain_preferences
  });
}

// Get user profile
export async function getUserProfile() {
  return apiRequest('/user/me');
  // Note: We're using the API_BASE_URL which is now '/api', 
  // so this will hit '/api/user/me'
}

// Get user recommendations
export async function getUserRecommendations(userId: string, domain: string, dataset: string, count: number = 10) {
  return apiRequest('/recommendations/user', 'POST', {
    user_id: userId,  // MUST be string format
    domain: domain,   // MUST match user preference
    dataset: dataset, // MUST match domain
    count: count
  });
}

// Get similar items
export async function getSimilarItems(itemId: string, domain: string, dataset: string, count: number = 5) {
  return apiRequest('/recommendations/similar', 'POST', {
    item_id: itemId,    // Can be string or number depending on domain
    domain: domain,     // MUST match item domain
    dataset: dataset,   // MUST match domain
    count: count
  });
}

// Get trending items
export async function getTrendingItems(domain: string, dataset: string, count: number = 10) {
  const userId = localStorage.getItem('user_id');
  
  return apiRequest('/recommendations/trending', 'POST', {
    user_id: userId,    // MUST be string format
    domain: domain,     // MUST match user preference
    dataset: dataset,   // MUST match domain
    count: count
  });
}

// Submit user rating
export async function submitRating(userId: string, itemId: string, rating: number, domain: string, dataset: string) {
  return apiRequest('/user/rate', 'POST', {
    user_id: userId,    // MUST be string
    item_id: itemId,    // Can be string or number depending on domain
    rating: rating,     // Numeric value (1-5)
    domain: domain,     // MUST match user preference
    dataset: dataset    // MUST match domain
  });
}
