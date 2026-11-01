import Link from "next/link";

export default function NotFound() {
  return (
    <div className="measure">
      <h1 className="text-[2rem]">No such page</h1>
      <p className="mt-3">
        This address is not one of the site's pages. <Link href="/" prefetch={false}>The marking</Link> is the place to start.
      </p>
    </div>
  );
}
