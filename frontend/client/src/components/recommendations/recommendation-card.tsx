import { Card, CardContent, CardHeader, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bookmark, ShoppingCart, Star } from "lucide-react";

interface RecommendationCardProps {
  item: {
    item_id: string;
    title: string;
    description: string;
    image?: string;
    rating: number;
    year?: string;
    genre?: string;
    price?: string;
    category?: string;
    score?: number;
    similarity?: number;
  };
  type: 'entertainment' | 'ecommerce';
}

export default function RecommendationCard({ item, type }: RecommendationCardProps) {
  // Placeholder for empty recommendations (should never show in production)
  if (!item) return null;
  
  // Fallback image
  const fallbackImage = type === 'entertainment' 
    ? "https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=600&h=340" 
    : "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&h=340";

  const image = item.image || fallbackImage;
  const title = item.title || (type === 'entertainment' ? 'Movie Title' : 'Product Name');
  const description = item.description || 'No description available';
  const rating = item.rating || 0;
  
  // Entertainment specific properties
  const year = item.year || 'N/A';
  const genre = item.genre || 'Unknown';
  
  // E-commerce specific properties
  const price = item.price || '£0.00';
  const category = item.category || 'Unknown';

  // Render star ratings
  const StarRating = () => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
    
    return (
      <div className="flex items-center">
        <div className="flex text-yellow-400">
          {[...Array(fullStars)].map((_, i) => (
            <Star key={`full-${i}`} className="h-4 w-4 fill-current" />
          ))}
          {hasHalfStar && <Star className="h-4 w-4 fill-current half-filled" />}
          {[...Array(emptyStars)].map((_, i) => (
            <Star key={`empty-${i}`} className="h-4 w-4 text-gray-300" />
          ))}
        </div>
        <span className="text-gray-600 text-sm ml-1">{rating.toFixed(1)}</span>
      </div>
    );
  };

  return (
    <Card className="overflow-hidden hover:shadow-md transition-shadow h-full">
      <img 
        src={image} 
        alt={title}
        className="w-full h-48 object-cover object-center" 
      />
      <CardHeader className="pb-2">
        <h3 className="text-lg font-semibold line-clamp-1">{title}</h3>
        <StarRating />
      </CardHeader>
      <CardContent className="pt-0">
        {type === 'ecommerce' && (
          <p className="text-sm text-gray-700 font-semibold mt-1">{price}</p>
        )}
        <p className="text-gray-600 text-sm line-clamp-2">{description}</p>
      </CardContent>
      <CardFooter className="flex justify-between items-center pt-0 mt-auto">
        <span className="text-sm text-gray-500">
          {type === 'entertainment' 
            ? `${year} • ${genre}` 
            : `${category}`
          }
        </span>
        <Button variant="ghost" size="sm">
          {type === 'entertainment' ? (
            <>
              <Bookmark className="mr-1 h-4 w-4" /> Save
            </>
          ) : (
            <>
              <ShoppingCart className="mr-1 h-4 w-4" /> View
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
