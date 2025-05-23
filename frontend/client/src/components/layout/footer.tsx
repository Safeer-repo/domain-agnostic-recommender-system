import { Star } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 py-6 mt-auto">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <span className="flex items-center text-gray-600">
              <Star className="h-4 w-4 text-primary mr-2" />
              <span className="font-medium">RecommendMe</span>
            </span>
            <p className="text-gray-500 text-sm mt-1">Your personal recommender for movies and products</p>
          </div>
          
          <div className="text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} RecommendMe. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
}
