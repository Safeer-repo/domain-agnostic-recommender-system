import { useState } from "react";
import { useAuth } from "@/contexts/auth-context";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { MenuIcon, Star } from "lucide-react";

export default function Navbar() {
  const { user, logout } = useAuth();
  const [_, navigate] = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/auth");
    setIsOpen(false);
  };

  return (
    <header className="bg-white shadow-sm">
      <nav className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Star className="h-6 w-6 text-primary" />
          <Link href="/">
            <span className="text-xl font-semibold text-primary cursor-pointer">
              RecommendMe
            </span>
          </Link>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex space-x-6 items-center">
          {user ? (
            <>
              <Link href="/dashboard">
                <span className="text-gray-600 hover:text-primary font-medium text-sm cursor-pointer">
                  Dashboard
                </span>
              </Link>
              <Link href="/profile">
                <span className="text-gray-600 hover:text-primary font-medium text-sm cursor-pointer">
                  Profile
                </span>
              </Link>
              <Button 
                variant="ghost" 
                onClick={handleLogout}
                className="text-gray-600 hover:text-primary font-medium text-sm"
              >
                Logout
              </Button>
            </>
          ) : (
            <>
              <Link href="/auth">
                <span className="text-gray-600 hover:text-primary font-medium text-sm cursor-pointer">
                  Login
                </span>
              </Link>
              <Button 
                variant="default" 
                size="sm"
                onClick={() => navigate("/auth")}
              >
                Register
              </Button>
            </>
          )}
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden">
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <MenuIcon className="h-6 w-6" />
                <span className="sr-only">Open menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>RecommendMe</SheetTitle>
                <SheetDescription>
                  Your personal recommender system
                </SheetDescription>
              </SheetHeader>
              <div className="flex flex-col space-y-4 mt-6">
                {user ? (
                  <>
                    <Button 
                      variant="ghost" 
                      className="justify-start" 
                      onClick={() => {
                        navigate("/dashboard");
                        setIsOpen(false);
                      }}
                    >
                      Dashboard
                    </Button>
                    <Button 
                      variant="ghost" 
                      className="justify-start" 
                      onClick={() => {
                        navigate("/profile");
                        setIsOpen(false);
                      }}
                    >
                      Profile
                    </Button>
                    <Button 
                      variant="ghost" 
                      className="justify-start" 
                      onClick={handleLogout}
                    >
                      Logout
                    </Button>
                  </>
                ) : (
                  <>
                    <Button 
                      variant="ghost" 
                      className="justify-start" 
                      onClick={() => {
                        navigate("/auth");
                        setIsOpen(false);
                      }}
                    >
                      Login
                    </Button>
                    <Button 
                      variant="default" 
                      onClick={() => {
                        navigate("/auth");
                        setIsOpen(false);
                      }}
                    >
                      Register
                    </Button>
                  </>
                )}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </header>
  );
}
