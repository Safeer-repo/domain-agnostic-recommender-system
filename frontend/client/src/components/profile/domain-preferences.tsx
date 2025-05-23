import { User } from "@shared/types";

interface DomainPreferencesProps {
  user: User;
}

export default function DomainPreferences({ user }: DomainPreferencesProps) {
  // Check domain preferences
  const hasEntertainment = user.domain_preferences?.entertainment?.includes("movielens");
  const hasEcommerce = user.domain_preferences?.ecommerce?.includes("amazon");

  return (
    <div className="space-y-4">
      <div className="flex items-center">
        <div className={`w-5 h-5 rounded-full border-2 border-gray-300 flex-shrink-0 ${hasEntertainment ? 'border-primary' : ''}`}>
          {hasEntertainment && (
            <div className="w-3 h-3 bg-primary rounded-full m-auto"></div>
          )}
        </div>
        <span className="ml-3 text-gray-700">Entertainment (Movies)</span>
      </div>
      
      <div className="flex items-center">
        <div className={`w-5 h-5 rounded-full border-2 border-gray-300 flex-shrink-0 ${hasEcommerce ? 'border-primary' : ''}`}>
          {hasEcommerce && (
            <div className="w-3 h-3 bg-primary rounded-full m-auto"></div>
          )}
        </div>
        <span className="ml-3 text-gray-700">E-commerce (Products)</span>
      </div>
    </div>
  );
}
