import { useEffect, useState } from "react";
import { getUserRecommendations, getSimilarItems, submitRating, getTrendingItems } from "@/lib/api";
import RecommendationCard from "@/components/recommendations/recommendation-card";
import { useAuth } from "@/contexts/auth-context";
import { Loader2, AlertCircle, Search, Star, TrendingUp, Info } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";

// Types for our recommendations
interface Recommendation {
  item_id: string;
  score?: number;
  similarity?: number;
  title: string;
  description: string;
  image?: string;
  year?: string;
  genre?: string;
  price?: string;
  category?: string;
  rating: number;
}

// Mock data for visualization
const MOCK_ITEMS = {
  entertainment: {
    names: [
      "The Shawshank Redemption", "The Godfather", "Pulp Fiction", 
      "The Dark Knight", "Schindler's List", "Forrest Gump",
      "The Matrix", "Goodfellas", "Interstellar", "Inception"
    ],
    images: [
      "https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1536440136628-849c177e76a1?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1478720568477-152d9b164e26?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1512070679279-8988d32161be?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1500099817043-86ae4e0ca04f?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1536440136628-849c177e76a1?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?auto=format&fit=crop&w=600&h=340"
    ],
    genres: ["Drama", "Crime", "Action", "Sci-Fi", "Thriller", "Comedy", "Romance", "Adventure"],
    years: ["2008", "1994", "1999", "2010", "2014", "2001", "1997", "2022", "2023"]
  },
  ecommerce: {
    names: [
      "Wireless Headphones", "Smart Watch", "Laptop", "Phone", 
      "Bluetooth Speaker", "Fitness Tracker", "Digital Camera", 
      "Tablet", "Gaming Console", "Smart Home Hub"
    ],
    images: [
      "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1546868871-7041f2a55e12?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1576633587382-13ddf37b1fc1?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1585790050230-5ab128623df7?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1486401899868-0e435ed85128?auto=format&fit=crop&w=600&h=340",
      "https://images.unsplash.com/photo-1558089687-f282ffcbc0d4?auto=format&fit=crop&w=600&h=340"
    ],
    categories: ["Electronics", "Wearables", "Computers", "Audio", "Mobile", "Gaming", "Smart Home", "Accessories"],
    prices: ["£49.99", "£149.99", "£999.99", "£79.99", "£199.99", "£249.99", "£599.99", "£19.99"]
  }
};

// Generate mock data based on item_id and domain
function generateMockItem(itemId: string, domain: string, score?: number, similarity?: number): Recommendation {
  const isEntertainment = domain === 'entertainment';
  const mockDomain = isEntertainment ? MOCK_ITEMS.entertainment : MOCK_ITEMS.ecommerce;
  
  // Ensure itemId is a string
  const itemIdStr = String(itemId);
  
  // Make a safe hash generation that works with various item_id formats
  let hash = 0;
  try {
    // Use a safer method to generate hash from item_id string
    if (itemIdStr) {
      for (let i = 0; i < itemIdStr.length; i++) {
        hash = ((hash << 5) - hash) + itemIdStr.charCodeAt(i);
        hash = hash & hash; // Convert to 32bit integer
      }
    }
    
    // Ensure we have a positive number
    hash = Math.abs(hash || Date.now());
  } catch (e) {
    // Fallback in case of any parsing issues
    hash = Date.now() % 1000;
  }
  
  const nameIndex = hash % mockDomain.names.length;
  const imageIndex = (hash * 2) % mockDomain.images.length;
  
  if (isEntertainment) {
    const genreIndex = hash % MOCK_ITEMS.entertainment.genres.length;
    const yearIndex = (hash * 3) % MOCK_ITEMS.entertainment.years.length;
    
    return {
      item_id: itemIdStr,
      score,
      similarity,
      title: mockDomain.names[nameIndex],
      description: `A captivating ${MOCK_ITEMS.entertainment.genres[genreIndex]} film from ${MOCK_ITEMS.entertainment.years[yearIndex]}`,
      image: mockDomain.images[imageIndex],
      year: MOCK_ITEMS.entertainment.years[yearIndex],
      genre: MOCK_ITEMS.entertainment.genres[genreIndex],
      rating: ((hash % 40) + 10) / 10 // Rating between 1.0 and 5.0
    };
  } else {
    const categoryIndex = hash % MOCK_ITEMS.ecommerce.categories.length;
    const priceIndex = (hash * 3) % MOCK_ITEMS.ecommerce.prices.length;
    
    return {
      item_id: itemIdStr,
      score,
      similarity,
      title: mockDomain.names[nameIndex],
      description: `Premium quality ${MOCK_ITEMS.ecommerce.categories[categoryIndex]} device with advanced features`,
      image: mockDomain.images[imageIndex],
      price: MOCK_ITEMS.ecommerce.prices[priceIndex],
      category: MOCK_ITEMS.ecommerce.categories[categoryIndex],
      rating: ((hash % 40) + 10) / 10 // Rating between 1.0 and 5.0
    };
  }
}

// This function has been replaced by loadTrendingItems which uses the API

export default function DashboardPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  
  const [currentDomain, setCurrentDomain] = useState<string>('entertainment');
  const [currentDataset, setCurrentDataset] = useState<string>('movielens');
  const [domainType, setDomainType] = useState<string>("Entertainment");
  
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [trendingItems, setTrendingItems] = useState<Recommendation[]>([]);
  const [similarItems, setSimilarItems] = useState<Recommendation[]>([]);
  
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingTrending, setLoadingTrending] = useState<boolean>(false);
  const [loadingSimilar, setLoadingSimilar] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [trendingError, setTrendingError] = useState<string>("");
  const [coldStartMessage, setColdStartMessage] = useState<string>("");
  const [ratingCount, setRatingCount] = useState<number>(0);
  
  const [selectedItem, setSelectedItem] = useState<Recommendation | null>(null);
  const [userRating, setUserRating] = useState<number>(0);

  // Extract domain preferences from user profile and start loading data immediately
  useEffect(() => {
    if (user) {
      let domain = 'entertainment';
      let dataset = 'movielens';
      let displayType = "Entertainment";

      if (user.domain_preferences) {
        // Check for entertainment domain
        if (user.domain_preferences.entertainment?.includes("movielens")) {
          domain = 'entertainment';
          dataset = 'movielens';
          displayType = "Entertainment";
        } 
        // Check for ecommerce domain
        else if (user.domain_preferences.ecommerce?.includes("amazon")) {
          domain = 'ecommerce';
          dataset = 'amazon';
          displayType = "E-commerce";
        }
      }

      console.log(`User domain: ${domain}, dataset: ${dataset}`);
      setCurrentDomain(domain);
      setCurrentDataset(dataset);
      setDomainType(displayType);

      // Load rating count from localStorage
      const storedRatingCount = parseInt(localStorage.getItem('ratingCount') || '0');
      setRatingCount(storedRatingCount);
      console.log(`User has rated ${storedRatingCount} items so far`);

      // Start parallel loading of recommendations and trending items
      // This helps reduce the perceived loading time
      const loadAllData = async () => {
        // Start loading both recommendations and trending items in parallel
        await Promise.all([
          loadRecommendations(domain, dataset),
          loadTrendingItems(domain, dataset)
        ]);
      };
      
      loadAllData();
    }
  }, [user]);

  // Load user recommendations
  const loadRecommendations = async (domain: string, dataset: string) => {
    if (!user) return;

    setLoading(true);
    setError("");
    setColdStartMessage("");

    try {
      const userId = user.user_id;
      console.log(`Loading recommendations for user ${userId} in domain ${domain}, dataset ${dataset}`);
      
      const response = await getUserRecommendations(userId, domain, dataset);
      
      if (response.error) {
        throw new Error(response.error.message);
      }
      
      // Check for cold start scenario
      if (response.data?.status === "cold_start") {
        console.log("Cold start detected:", response.data.message);
        
        if (ratingCount >= 10) {
          // If our local count is 10+ but server says cold start, 
          // show a more specific message that the backend is still processing
          setColdStartMessage("Your ratings are being processed. Keep rating more items to improve recommendations.");
        } else {
          // Normal cold start handling for fewer than 10 ratings
          setColdStartMessage(response.data.message || "Please rate some items to get personalized recommendations");
        }
        
        setRecommendations([]);
        return;
      }
      
      // Transform the API response into our recommendation format
      const recommendationData = response.data?.recommendations || [];
      
      // Debug log to check the structure of the response
      console.log('Recommendation data:', recommendationData);
      
      if (!Array.isArray(recommendationData)) {
        throw new Error('Invalid recommendation data format');
      }
      
      // Process recommendations based on domain format
      const enhancedRecommendations: Recommendation[] = recommendationData.map((item: any) => {
        // Ensure item_id exists and convert to string if it's a number
        const itemId = typeof item.item_id === 'number' ? 
          item.item_id.toString() : item.item_id || `unknown-${Math.random().toString(36).substring(7)}`;
          
        return generateMockItem(itemId, domain, item.score);
      });
      
      setRecommendations(enhancedRecommendations);
    } catch (error: any) {
      console.error('Failed to load recommendations:', error);
      setError(error?.message || "Failed to load recommendations. Please try again later.");
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  };

  // Load trending items from API
  const loadTrendingItems = async (domain: string, dataset: string) => {
    if (!user) return;
    
    setLoadingTrending(true);
    setTrendingError("");
    
    try {
      console.log(`Loading trending items for domain ${domain}, dataset ${dataset}`);
      
      const response = await getTrendingItems(domain, dataset);
      
      if (response.error) {
        throw new Error(response.error.message);
      }
      
      // Get the trending items from the response
      const trendingItemsData = response.data?.trending_items || [];
      
      // Debug log to check the structure of the response
      console.log('Trending items data:', trendingItemsData);
      
      if (!Array.isArray(trendingItemsData)) {
        throw new Error('Invalid trending items data format');
      }
      
      // Process trending items data
      const enhancedTrendingItems: Recommendation[] = trendingItemsData.map((item: any) => {
        // Ensure item_id exists and convert to string if it's a number
        const itemId = typeof item.item_id === 'number' ? 
          item.item_id.toString() : item.item_id || `unknown-${Math.random().toString(36).substring(7)}`;
          
        return generateMockItem(itemId, domain, item.score);
      });
      
      setTrendingItems(enhancedTrendingItems);
    } catch (error: any) {
      console.error('Failed to load trending items:', error);
      setTrendingError(error?.message || "Failed to load trending items");
      
      // Fall back to mock data if API fails
      const mockTrending = generateMockTrendingItems(domain);
      setTrendingItems(mockTrending);
    } finally {
      setLoadingTrending(false);
    }
  };
  
  // Generate mock trending items (fallback function)
  function generateMockTrendingItems(domain: string, count: number = 5): Recommendation[] {
    return Array.from({ length: count }, (_, i) => {
      const trendingId = `trending-${domain}-${i}-${Date.now().toString().substring(9)}`;
      return generateMockItem(trendingId, domain);
    });
  };

  // Load similar items
  const loadSimilarItems = async (itemId: string) => {
    if (!user) return;
    
    setLoadingSimilar(true);
    
    try {
      const response = await getSimilarItems(itemId, currentDomain, currentDataset);
      
      if (response.error) {
        throw new Error(response.error.message);
      }
      
      // Transform the API response into our recommendation format
      const similarItemsData = response.data?.similar_items || [];
      
      // Debug log to check the structure of the response
      console.log('Similar items data:', similarItemsData);
      
      if (!Array.isArray(similarItemsData)) {
        throw new Error('Invalid similar items data format');
      }
      
      // Process similar items with the same careful handling of item_id
      const enhancedSimilarItems: Recommendation[] = similarItemsData.map((item: any) => {
        // Ensure item_id exists and convert to string if it's a number
        const itemId = typeof item.item_id === 'number' ? 
          item.item_id.toString() : item.item_id || `unknown-${Math.random().toString(36).substring(7)}`;
          
        return generateMockItem(itemId, currentDomain, undefined, item.similarity);
      });
      
      setSimilarItems(enhancedSimilarItems);
    } catch (error: any) {
      console.error('Failed to load similar items:', error);
      toast({
        title: "Error",
        description: error?.message || "Failed to load similar items. Please try again.",
        variant: "destructive"
      });
      setSimilarItems([]);
    } finally {
      setLoadingSimilar(false);
    }
  };

  // Handle item click
  const handleItemClick = (item: Recommendation) => {
    setSelectedItem(item);
    setUserRating(0);
    loadSimilarItems(item.item_id);
  };

  // Submit user rating
  const handleRateItem = async () => {
    if (!user || !selectedItem || userRating === 0) return;
    
    try {
      const userId = user.user_id;
      
      console.log(`Submitting rating for item ${selectedItem.item_id} with rating ${userRating}`);
      
      const response = await submitRating(
        userId, 
        selectedItem.item_id, 
        userRating, 
        currentDomain, 
        currentDataset
      );
      
      console.log('Rating submission response:', response);
      
      if (response.error) {
        throw new Error(response.error.message);
      }
      
      // Increment and save the rating count in localStorage
      const currentRatingCount = parseInt(localStorage.getItem('ratingCount') || '0');
      const newRatingCount = currentRatingCount + 1;
      localStorage.setItem('ratingCount', newRatingCount.toString());
      console.log(`User has now rated ${newRatingCount} items`);
      
      // Update the state to reflect the new count
      setRatingCount(newRatingCount);
      
      toast({
        title: "Rating Submitted",
        description: "Your rating has been submitted successfully.",
      });
      
      // Close the dialog after successful rating
      setSelectedItem(null);
      
      // Add a small delay before refreshing recommendations to allow backend to process
      setTimeout(() => {
        loadRecommendations(currentDomain, currentDataset);
      }, 500);
    } catch (error: any) {
      console.error('Failed to submit rating:', error);
      toast({
        title: "Error",
        description: error?.message || "Failed to submit your rating. Please try again.",
        variant: "destructive"
      });
    }
  };

  return (
    <div>
      {/* Welcome header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-blue-700 bg-clip-text text-transparent">
          Welcome, {user?.username}
        </h1>
        <p className="text-muted-foreground">
          Here are your personalised recommendations based on your preferences.
        </p>
      </div>
      
      {/* Domain indicator with rating counter reset option */}
      <div className="bg-primary/10 text-primary px-4 py-3 rounded-lg mb-6 flex items-center justify-between">
        <span>Showing recommendations for your preferred domain: <span className="font-semibold">{domainType}</span></span>
        <div className="flex items-center gap-2">
          <span className="text-sm">Rating count: {ratingCount}</span>
          <Button 
            variant="outline" 
            size="sm"
            className="text-xs"
            onClick={() => {
              localStorage.setItem('ratingCount', '0');
              setRatingCount(0);
              toast({
                title: "Ratings Reset",
                description: "Your rating count has been reset for testing purposes.",
              });
            }}
          >
            Reset Ratings
          </Button>
        </div>
      </div>
      
      {/* Top Picks Section */}
      <div className="mb-10">
        <div className="flex items-center mb-4">
          <Star className="mr-2 h-5 w-5 text-primary" />
          <h2 className="text-2xl font-semibold">Top Picks for You</h2>
        </div>
        
        {/* Loading state */}
        {loading && (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </div>
        )}
        
        {/* Error state */}
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error}
            </AlertDescription>
          </Alert>
        )}
        
        {/* Cold Start state with progress indicator */}
        {!loading && !error && coldStartMessage && (
          <Alert className="mb-6 bg-muted/40 border-primary/20">
            <Info className="h-4 w-4 text-primary" />
            <AlertDescription className="flex flex-col gap-2">
              <span>{coldStartMessage}</span>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Rating progress</span>
                  <span className="font-medium">{ratingCount}/10</span>
                </div>
                <Progress value={ratingCount * 10} max={100} className="h-2" />
              </div>
            </AlertDescription>
          </Alert>
        )}
        
        {/* Empty state - only show if not in cold start */}
        {!loading && !error && !coldStartMessage && recommendations.length === 0 && (
          <div className="text-center py-12 bg-muted/20 rounded-lg">
            <div className="bg-muted mx-auto rounded-full w-16 h-16 flex items-center justify-center mb-4">
              <Search className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-medium text-gray-700 mb-2">No Recommendations Yet</h3>
            <p className="text-muted-foreground">We're working on finding the perfect recommendations for you.</p>
          </div>
        )}
        
        {/* Recommendations grid */}
        {!loading && !error && recommendations.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {recommendations.map((item) => (
              <div key={item.item_id} onClick={() => handleItemClick(item)} className="cursor-pointer">
                <RecommendationCard 
                  item={item} 
                  type={currentDomain as 'entertainment' | 'ecommerce'} 
                />
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Trending Now Section */}
      <div className="mb-10">
        <div className="flex items-center mb-4">
          <TrendingUp className="mr-2 h-5 w-5 text-primary" />
          <h2 className="text-2xl font-semibold">Trending Now</h2>
        </div>
        
        {/* Loading state */}
        {loadingTrending && (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </div>
        )}
        
        {/* Error state */}
        {trendingError && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {trendingError}
            </AlertDescription>
          </Alert>
        )}
        
        {/* Empty state */}
        {!loadingTrending && !trendingError && trendingItems.length === 0 && (
          <div className="text-center py-12 bg-muted/20 rounded-lg">
            <div className="bg-muted mx-auto rounded-full w-16 h-16 flex items-center justify-center mb-4">
              <Search className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-medium text-gray-700 mb-2">No Trending Items</h3>
            <p className="text-muted-foreground">We couldn't find any trending items at the moment.</p>
          </div>
        )}
        
        {/* Trending items grid */}
        {!loadingTrending && !trendingError && trendingItems.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {trendingItems.map((item) => (
              <div key={item.item_id} onClick={() => handleItemClick(item)} className="cursor-pointer">
                <RecommendationCard 
                  item={item} 
                  type={currentDomain as 'entertainment' | 'ecommerce'} 
                />
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Item Details Dialog */}
      <Dialog open={!!selectedItem} onOpenChange={(open) => !open && setSelectedItem(null)}>
        <DialogContent className="max-w-4xl">
          {selectedItem && (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl">{selectedItem.title}</DialogTitle>
                <DialogDescription>
                  {currentDomain === 'entertainment' ? 
                    `${selectedItem.year} • ${selectedItem.genre}` : 
                    `${selectedItem.category} • ${selectedItem.price}`}
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-4">
                <div>
                  <img 
                    src={selectedItem.image} 
                    alt={selectedItem.title}
                    className="w-full h-64 object-cover rounded-lg shadow-md"
                  />
                </div>
                
                <div className="space-y-4">
                  <p className="text-muted-foreground">{selectedItem.description}</p>
                  
                  {/* Rating interface */}
                  <div className="space-y-2">
                    <h3 className="font-medium">Rate this {currentDomain === 'entertainment' ? 'movie' : 'product'}</h3>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <button
                          key={rating}
                          className={`p-1 rounded-full ${userRating >= rating ? 'text-yellow-400' : 'text-gray-300'}`}
                          onClick={() => setUserRating(rating)}
                        >
                          <Star className="h-8 w-8" fill={userRating >= rating ? 'currentColor' : 'none'} />
                        </button>
                      ))}
                    </div>
                    <Button 
                      onClick={handleRateItem}
                      disabled={userRating === 0}
                      className="mt-2"
                    >
                      Submit Rating
                    </Button>
                  </div>
                </div>
              </div>
              
              {/* Similar Items Section */}
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-4">Similar Items You Might Like</h3>
                
                {loadingSimilar ? (
                  <div className="flex justify-center py-6">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : similarItems.length > 0 ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {similarItems.map((item) => (
                      <div key={item.item_id} onClick={() => handleItemClick(item)} className="cursor-pointer">
                        <div className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow">
                          <img 
                            src={item.image} 
                            alt={item.title} 
                            className="w-full h-32 object-cover"
                          />
                          <div className="p-2">
                            <h4 className="font-medium text-sm truncate">{item.title}</h4>
                            <div className="flex items-center mt-1">
                              <Star className="h-3 w-3 text-yellow-400 fill-current" />
                              <span className="text-xs text-gray-600 ml-1">{item.rating.toFixed(1)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-4">No similar items found.</p>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
