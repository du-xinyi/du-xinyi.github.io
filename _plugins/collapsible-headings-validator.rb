# frozen_string_literal: true

module CollapsibleHeadingsValidator
  SUPPORTED_LEVELS = [2, 3, 4].freeze

  module_function

  def validate(item)
    levels = item.data['collapsible_headings']
    expanded = item.data['expanded_headings']
    return if levels.nil? && expanded.nil?

    validate_levels(item, 'collapsible_headings', levels)

    return if expanded.nil?

    validate_levels(item, 'expanded_headings', expanded)

    invalid_expanded = expanded - levels
    return if invalid_expanded.empty?

    fail_build(
      item,
      "expanded_headings contains levels that are not collapsible: #{invalid_expanded.join(', ')}"
    )
  end

  def validate_levels(item, key, levels)
    valid = levels.is_a?(Array) &&
            levels.uniq.length == levels.length &&
            levels.all? { |level| SUPPORTED_LEVELS.include?(level) }
    return if valid

    fail_build(item, "#{key} must be a unique array containing only 2, 3, and 4")
  end

  def fail_build(item, message)
    path = item.respond_to?(:relative_path) ? item.relative_path : item.path
    raise Jekyll::Errors::FatalException, "#{path}: #{message}"
  end
end

Jekyll::Hooks.register :documents, :pre_render do |document|
  CollapsibleHeadingsValidator.validate(document)
end

Jekyll::Hooks.register :pages, :pre_render do |page|
  CollapsibleHeadingsValidator.validate(page)
end
