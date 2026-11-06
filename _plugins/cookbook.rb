# frozen_string_literal: true

# Assemble the cookbook post from the `recipes` collection.
#
# Recipes live in `_recipes/<category>/[<group>/]<slug>.md`; categories and
# their order are declared in `_data/cookbook.yml`. Within a category, recipes
# are sorted by an optional `order` front matter value first, then by filename.
#
# This hook attaches the sorted recipe documents to each category/group in
# `site.data.cookbook`, keeps each recipe's raw Markdown in `markdown` (Jekyll
# also renders `output: false` documents, so `content` may already be HTML
# depending on render order), and sets the cookbook post's
# `last_modified_at` from the latest commit touching any recipe.
module Cookbook
  module_function

  def fail_build(message)
    raise Jekyll::Errors::FatalException, "Cookbook: #{message}"
  end

  def sort_key(doc)
    order = doc.data['order']
    [order.nil? ? 1 : 0, order.to_i, doc.basename_without_ext]
  end

  def attach(node, prefix, docs_by_dir, declared)
    dir = prefix.empty? ? node['id'].to_s : File.join(prefix, node['id'].to_s)
    declared << dir
    docs = docs_by_dir.fetch(dir, [])

    if node['groups']
      fail_build("#{dir}/ has groups, so recipes must go in a group directory") unless docs.empty?
      node['groups'].each { |group| attach(group, dir, docs_by_dir, declared) }
    else
      node['recipes'] = docs.sort_by { |doc| sort_key(doc) }
    end
  end

  def build(site)
    collection = site.collections['recipes']
    categories = site.data['cookbook']
    return if collection.nil? || categories.nil?

    docs_by_dir = collection.docs.group_by do |doc|
      doc.data['markdown'] = doc.content
      File.dirname(doc.relative_path.delete_prefix('_recipes/'))
    end

    declared = []
    categories.each { |category| attach(category, '', docs_by_dir, declared) }

    undeclared = docs_by_dir.keys - declared
    return if undeclared.empty?

    fail_build("directories not declared in _data/cookbook.yml: #{undeclared.join(', ')}")
  end

  def update_lastmod(site)
    paths = [File.join(site.source, '_recipes'), File.join(site.source, '_data', 'cookbook.yml')]

    site.posts.docs.each do |post|
      next unless post.data['cookbook']

      args = ([post.path] + paths).map { |path| "\"#{path}\"" }.join(' ')
      date = `git log -1 --pretty="%ad" --date=iso -- #{args}`.strip
      post.data['last_modified_at'] = date unless date.empty?
    end
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  Cookbook.build(site)
  Cookbook.update_lastmod(site)
end
