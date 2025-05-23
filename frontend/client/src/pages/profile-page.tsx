import { useAuth } from "@/contexts/auth-context";
import DomainPreferences from "@/components/profile/domain-preferences";
import { User } from "lucide-react";
import { formatDistance } from "date-fns";

export default function ProfilePage() {
  const { user } = useAuth();

  if (!user) {
    return null;
  }

  // Format dates for display
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  return (
    <div>
      <h1 className="text-3xl font-semibold mb-6">Your Profile</h1>
      
      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center space-x-4">
            <div className="bg-primary-100 text-primary-700 rounded-full p-4">
              <User className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">{user.username}</h2>
              <p className="text-gray-600">{user.email}</p>
            </div>
          </div>
        </div>
        
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4">Domain Preferences</h3>
          
          <DomainPreferences user={user} />
          
          <div className="mt-8">
            <h3 className="text-lg font-semibold mb-4">Account Information</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Member Since</span>
                <span className="font-medium">
                  {user.created_at ? formatTimestamp(user.created_at) : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Last Updated</span>
                <span className="font-medium">
                  {user.last_updated ? formatTimestamp(user.last_updated) : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
