local seen_folie = false

function Header(el)
  local text = pandoc.utils.stringify(el)
  if el.level == 1 and text == "Notizen und Begriffserklärungen" then
    return {pandoc.RawBlock("latex", "\\newpage"), el}
  end
  if el.level == 2 and text:match("^Folie") then
    seen_folie = true
    return {pandoc.RawBlock("latex", "\\newpage"), el}
  end
  return el
end
