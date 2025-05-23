import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import LoginForm from "@/components/auth/login-form";
import RegisterForm from "@/components/auth/register-form";
import { useAuth } from "@/contexts/auth-context";
import { Redirect } from "wouter";

export default function AuthPage() {
  const { user, isLoading } = useAuth();

  // Redirect to dashboard if already logged in
  if (user && !isLoading) {
    return <Redirect to="/dashboard" />;
  }

  return (
    <div className="flex flex-col md:flex-row md:items-center gap-8 max-w-6xl mx-auto">
      {/* Left column - Auth forms */}
      <div className="md:w-1/2">
        <Card className="w-full">
          <CardContent className="pt-6">
            <Tabs defaultValue="login">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="login">Login</TabsTrigger>
                <TabsTrigger value="register">Register</TabsTrigger>
              </TabsList>
              
              <TabsContent value="login" className="mt-4">
                <LoginForm />
              </TabsContent>
              
              <TabsContent value="register" className="mt-4">
                <RegisterForm />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Right column - Hero section */}
      <div className="md:w-1/2">
        <div className="space-y-4">
          <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-primary to-blue-700 bg-clip-text text-transparent">
            Personalised Recommendations
          </h1>
          <p className="text-lg text-muted-foreground">
            Discover content and products tailored specifically to your preferences.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="text-primary-600 mb-2">
                <i className="fa fa-film text-2xl"></i>
              </div>
              <h3 className="font-semibold mb-1">Entertainment</h3>
              <p className="text-sm text-gray-600">Get movie recommendations based on your taste</p>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="text-primary-600 mb-2">
                <i className="fa fa-shopping-cart text-2xl"></i>
              </div>
              <h3 className="font-semibold mb-1">E-commerce</h3>
              <p className="text-sm text-gray-600">Discover products you'll love</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
