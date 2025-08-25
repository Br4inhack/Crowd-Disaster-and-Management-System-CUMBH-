// app/page.tsx
import Link from 'next/link';

export default function AuthoritySelection() {
  const roles = ['Event Management', 'Security', 'Emergency Services'];

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8">Select Your Authority</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {roles.map((role) => (
          <Link
            key={role}
            href={`/dashboard?role=${encodeURIComponent(role)}`}
            className="p-8 bg-gray-800 rounded-lg text-center hover:bg-indigo-600 transition-colors"
          >
            <span className="text-2xl">{role}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}