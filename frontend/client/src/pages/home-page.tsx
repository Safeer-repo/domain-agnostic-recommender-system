import { useAuth } from "@/contexts/auth-context";
import { Button } from "@/components/ui/button";
import { Redirect, useLocation } from "wouter";

export default function HomePage() {
  const { user } = useAuth();
  const [_, navigate] = useLocation();

  // Redirect to dashboard if logged in
  if (user) {
    return <Redirect to="/dashboard" />;
  }

  return (
    <div className="flex flex-col items-center text-center max-w-3xl mx-auto py-12">
      <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary to-blue-700 bg-clip-text text-transparent mb-6">
        Personalised Recommendations
      </h1>
      
      <p className="text-xl text-muted-foreground mb-8">
        Discover content and products tailored specifically to your preferences with RecommendMe.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4">
        <Button 
          size="lg" 
          onClick={() => navigate("/auth")}
          className="px-8"
        >
          Get Started
        </Button>
        
        <Button 
          size="lg" 
          variant="outline" 
          onClick={() => navigate("/auth")}
          className="px-8"
        >
          Learn More
        </Button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-16">
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <div className="text-primary-600 mb-4 text-3xl">
            <i className="fas fa-film"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Entertainment Recommendations</h3>
          <p className="text-muted-foreground">
            Get personalized movie recommendations based on your taste and viewing history.
          </p>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <div className="text-primary-600 mb-4 text-3xl">
            <i className="fas fa-shopping-cart"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">E-commerce Recommendations</h3>
          <p className="text-muted-foreground">
            Discover products you'll love with our smart recommendation engine.
          </p>
        </div>
      </div>
    </div>
  );
}
